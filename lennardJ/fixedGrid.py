import numpy as np
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from IPython import embed

from plot_episode_summaries import *
from LJEnvironment import *
from LJNetwork import Qnetwork



def softmax(x):
    """
    
    Arguments:
    - `x`:
    """

    c = max(x)
    tmp = c + np.log(np.sum(np.exp(x-c)))
    # res = np.exp(x)
    return np.exp(x-tmp)# res/np.sum(res)


def get_all_points_not_in_gridlist(all_points,gridlist):
    """
    
    Arguments:
    - `all_points`:
    - `gridlist`:
    """
    slist = []
    candidate_feature_list = []
    for s,candidate in enumerate(all_points):
        if candidate.tolist() not in gridlist.tolist():
            candidate_gridlist = np.vstack((gridlist,np.array(candidate)))
            candidate_feature_list.append(LJEnv.getFeature(np.array([LJEnv.gridToXY(grid) for grid in candidate_gridlist])))
            slist.append(s)
        s += 1

    return slist,np.array(candidate_feature_list)

    

N_atoms = 10
max_n_episodes = 160
gamma = 0.99

r0 = 1.0
eps = 0
sigma= 1
params = (r0,eps,sigma)


LJEnv = LJEnvironment(params)


Q,CurrentFeature,NextFeature  = Qnetwork()

Qnext = tf.placeholder(tf.float32,shape=[None,1])
loss = tf.reduce_mean(tf.square(Q-Qnext))
trainer = tf.train.AdamOptimizer()
trainOp = trainer.minimize(loss)



gridlist = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [-1,0],
                     [-1,1],
                     [0,-1],
                     [1,-1]])


E = LJEnv.getEnergy(gridlist)
Currentfeat = LJEnv.getFeature(gridlist)

Nextgridlist = np.vstack((gridlist,np.array([-2,2])))
nextFeat = LJEnv.getFeature(Nextgridlist)



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Qout = sess.run(Q,feed_dict={CurrentFeature: Currentfeat.reshape((1,100)),
#                              NextFeature: nextFeat.reshape((1,100))})

all_points = []
# shear = np.array([[1,-1],[0,1]])
for x in range(-5,6):
    for y in range(-5,6):
        # all_points.append(np.dot(shear,np.array([x,y])))
        all_points.append([x,y])

        
all_points = np.array(all_points)

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(all_points.T[0],all_points.T[1])
# ax.set_xlim([-10,10])
# ax.set_ylim([-10,10])
# fig.show()
# stop

n_episodes = 0
QtargetList = []
CurrentFeatList = []
NewFeatList = []
Elist = []
running_mean = []

molecule_build_list = []
feature_build_list = []
slist_build_list = []
probs_build_list = []
episode_summary_list = []

episode_probe = [1,21,41,61,81,101,121,141,max_n_episodes+1]
probe_count = 0
while n_episodes < max_n_episodes:

    # Start with a single atom at the center
    gridlist = np.array([[0,0]])

    if np.mod(n_episodes,20) == 0 and n_episodes != 0:
        QtargetList = np.array(QtargetList).reshape((len(QtargetList),1))
        CurrentFeatList = np.array(CurrentFeatList)
        NewFeatList = np.array(NewFeatList)
        print('Training!')
        m = 0
        while m < 5:
            sess.run(trainOp,feed_dict={CurrentFeature: CurrentFeatList,
                                        NextFeature: NewFeatList,
                                        Qnext: QtargetList})
            m += 1
        print('Training done!')
        QtargetList = []
        CurrentFeatList = []
        NewFeatList = []
        
    for n in range(N_atoms-1):
        master_tic = time.time()
        # Get feature for current gridlist
        tic = time.time()

        CurrentFeat = LJEnv.getFeature(np.array([LJEnv.gridToXY(grid) for grid in gridlist]))
        feature_toc = time.time() - tic
        
        # Run all possible (resonable) positions through the Qnetwork
        tic = time.time()
        slist,CandidateFeatBatch = get_all_points_not_in_gridlist(all_points,gridlist)
        slist_toc = time.time()-tic
        # Copy CurrentFeat once for every candidate

        tic = time.time()
        CurrentFeatBatch = CurrentFeat
        for i in range(len(slist)-1):
            CurrentFeatBatch = np.vstack((CurrentFeatBatch,CurrentFeat))
        stack_toc = time.time() - tic

        # Get Q values for all candidates
        tic = time.time()
        Qlist = sess.run(Q,feed_dict={CurrentFeature: CurrentFeatBatch,
                                      NextFeature: CandidateFeatBatch})
        Qlisttoc = time.time() - tic

        # Turn Q values into probabilities for sampling
        # probs = (Qlist-np.min(Qlist))/np.sum(Qlist-np.min(Qlist))
        probs = softmax(Qlist)

        if n_episodes==episode_probe[probe_count]:
            molecule_build_list.append(gridlist)
            feature_build_list.append(CurrentFeat)
            slist_build_list.append(slist)
            probs_build_list.append(probs)
        
        # Sample new position according to Q values. slist contains indexes of
        # all_positions that are not in gridlist, i.e. no two atoms can be on top
        # of each other
        nextPointIndex = np.random.choice(slist,p=probs.flatten())
        nextPoint = all_points[nextPointIndex]

        # Add new point to gridlist and get the new feature
        gridlist = np.vstack((gridlist,nextPoint))
        NewFeat = LJEnv.getFeature(np.array([LJEnv.gridToXY(grid) for grid in gridlist]))

        # Run all positions through Qnetwork again, but now with the new gridlist. This
        # is used to get Qtarget.

        tic = time.time()
        slist,CandidateFeatBatch = get_all_points_not_in_gridlist(all_points,gridlist)
        slist_toc2 = time.time()-tic
        # Copy CurrentFeat once for every candidate
        NewFeatBatch = NewFeat
        for i in range(len(slist)-1):
            NewFeatBatch = np.vstack((NewFeatBatch,NewFeat))


        # Get new Q list
        NewQlist = sess.run(Q,feed_dict={CurrentFeature: NewFeatBatch,
                                         NextFeature: CandidateFeatBatch})
        
        # If all atoms are placed, calculate the energy and set the negative to reward
        if n == N_atoms-2:
            E =  LJEnv.getEnergy(np.array([LJEnv.gridToXY(grid) for grid in gridlist]))
            r = -np.power(E,3)
            Elist.append(E)
            if E == min(Elist):
                best_E = E
                best_grid = gridlist

            if n_episodes==episode_probe[probe_count]:
                molecule_build_list.append(gridlist)
                feature_build_list.append(NewFeat)

        else:
            r = 0
            
        # Qtarget update rule
        Qtarget = r + gamma*np.max(NewQlist)

        # Save current and new feature and Qtarget for batch training
        CurrentFeatList.append(CurrentFeat)
        NewFeatList.append(NewFeat)        
        QtargetList.append(Qtarget)
        master_toc = time.time() - master_tic

    if n_episodes > 20:
        running_mean.append(np.mean(Elist[n_episodes-20:]))

    if n_episodes==episode_probe[probe_count]:
        
        episode_summary_list.append([molecule_build_list,
                                     feature_build_list,
                                     slist_build_list,
                                     probs_build_list])
        
        
        molecule_build_list = []
        feature_build_list = []
        slist_build_list = []
        probs_build_list = []
        probe_count +=1 
        

    print('Episode: %i/%i \t Current energy: %4.4f \t Best energy = %4.4f' %(n_episodes,max_n_episodes,
                                                                             E,
                                                                             best_E))
    n_episodes+=1

    
        
        

        
plot_episode_summaries(episode_summary_list,all_points,N_atoms,LJEnv,Elist,'test')

        
stop

xylist = np.array([LJEnv.gridToXY(grid) for grid in best_grid])
all_points_XY = np.array([LJEnv.gridToXY(point) for point in all_points])





fig = plt.figure()
ax = fig.gca()
ax.set_xlim([-7,7])
ax.set_ylim([-7,7])
ax.plot(all_points_XY.T[0],all_points_XY.T[1],'go',alpha=0.5)
ax.plot(xylist.T[0],xylist.T[1],'bo')


fig.savefig('gridPlot_5_atoms')

fig = plt.figure()
ax = fig.gca()
ax.plot(Elist)
ax.plot(np.array(range(max_n_episodes-21))+21,running_mean)
fig.savefig('learning_curve_5_atoms')




fig,ax = plt.subplots(N_atoms,2,figsize = (4,4/2*N_atoms))


# molecule_build_list = np.array(molecule_build_list)
# feature_build_list = np.array(feature_build_list)
# slist_build_list = np.array(slist_build_list)
# probs_build_list = np.array(probs_build_list)


molecule_axes = ax.T[0]
feature_axes = ax.T[1]


for i,axis in enumerate(molecule_axes):
    axis.set_xlim([-8,8])
    axis.set_ylim([-8,8])


    if i<N_atoms-1:
        sp = np.array([LJEnv.gridToXY(point) for point in all_points[slist_build_list[i]]])
        # Create colors from probs list
        c = []
        max_p = np.max(probs_build_list[i])

        
        for p in probs_build_list[i]:
            col = 0.8*(1-p)
            c.append([col,col,col])
        axis.scatter(sp.T[0],sp.T[1],c=np.array(c).reshape((len(sp),3)))
        
    
    xylist = np.array([LJEnv.gridToXY(grid) for grid in molecule_build_list[i]])
    axis.scatter(xylist.T[0],xylist.T[1])

max_feat = np.max(feature_build_list)
    
for i,axis in enumerate(feature_axes):
    axis.set_ylim([0,max_feat])
    axis.plot(feature_build_list[i],'r')
    




fig.tight_layout()
fig.savefig('structure_feature_%i_atoms_%i_episodes' %(N_atoms,episode_probe),dpi=400)

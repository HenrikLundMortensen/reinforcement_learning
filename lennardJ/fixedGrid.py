import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import embed

from LJEnvironment import *
from LJNetwork import Qnetwork



def softmax(x):
    """
    
    Arguments:
    - `x`:
    """
    res = np.exp(x)
    return res/np.sum(res)
    

N_atoms = 7
max_n_episodes = 10
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
for x in range(-7,8):
    for y in range(-7,8):
        all_points.append([x,y])
        
all_points = np.array(all_points)

n_episodes = 0
QtargetList = []
CurrentFeatList = []
NewFeatList = []
rlist = []
while n_episodes < max_n_episodes:

    # Start with a single atom at the center
    gridlist = np.array([[0,0]])

    if np.mod(n_episodes,2) == 0 and n_episodes != 0:
        QtargetList = np.array(QtargetList).reshape((len(QtargetList),1))
        CurrentFeatList = np.array(CurrentFeatList)
        NewFeatList = np.array(NewFeatList)

        m = 0
        while m < 100:
            sess.run(trainOp,feed_dict={CurrentFeature: CurrentFeatList,
                                        NextFeature: NewFeatList,
                                        Qnext: QtargetList})
            m += 1

        QtargetList = []
        CurrentFeatList = []
        NewFeatList = []
        
    print(n_episodes)
    for n in range(N_atoms):
        # Get feature for current gridlist
        CurrentFeat = LJEnv.getFeature(gridlist)

        
        # Run all possible (resonable) positions through the Qnetwork
        s = 0
        slist = []
        Qlist = []
        for candidate in all_points:
            if candidate.tolist() not in gridlist.tolist():
                slist.append(s)
                candidate_gridlist = np.vstack((gridlist,np.array(candidate)))
                candidate_feature = LJEnv.getFeature(candidate_gridlist)
                Qlist.append(sess.run(Q,feed_dict={CurrentFeature: Currentfeat.reshape((1,100)),
                                                   NextFeature: candidate_feature.reshape((1,100))}))
            s += 1

        # Turn Q values into probabilities for sampling
        probs = softmax(Qlist)

        # Sample new position according to Q values. slist contains indexes of
        # all_positions that are not in gridlist, i.e. no two atoms can be on top
        # of each other
        nextPointIndex = np.random.choice(slist,p=probs.flatten())
        nextPoint = all_points[nextPointIndex]

        # Add new point to gridlist and get the new feature
        gridlist = np.vstack((gridlist,nextPoint))
        NewFeature = LJEnv.getFeature(gridlist)

        # Run all positions through Qnetwork again, but now with the new gridlist. This
        # is used to get Qtarget. 
        s = 0
        slist = []
        Qnextlist = []        
        for candidate in all_points:
            if candidate.tolist() not in gridlist.tolist():
                slist.append(s)
                candidate_gridlist = np.vstack((gridlist,np.array(candidate)))
                candidate_feature = LJEnv.getFeature(candidate_gridlist)
                Qnextlist.append(sess.run(Q,feed_dict={CurrentFeature: NewFeature.reshape((1,100)),
                                                   NextFeature: candidate_feature.reshape((1,100))}))
            s += 1

        # If all atoms are placed, calculate the energy and set the negative to reward
        if n == N_atoms-1:
            r = -LJEnv.getEnergy(gridlist)
            rlist.append(r)
        else:
            r = 0
            
        # Qtarget update rule
        Qtarget = r + gamma*np.max(Qnextlist)

        # Save current and new feature and Qtarget for batch training
        CurrentFeatList.append(Currentfeat)
        NewFeatList.append(NewFeature)        
        QtargetList.append(Qtarget)


    n_episodes+=1
            
        
        

        


        
# Qlist = []


xylist = np.array([LJEnv.gridToXY(grid) for grid in gridlist])
# xylist = np.array([LJEnv.gridToXY(point) for point in all_points])



















fig = plt.figure()
ax = fig.gca()
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.plot(xylist.T[0],xylist.T[1],'bo')

fig.savefig('gridPlot')


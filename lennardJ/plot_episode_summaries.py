from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed
import numpy as np

def plot_episode_summaries(eps_sum,all_points,N_atoms,LJEnv,Elist_list,name):
    """

    """
    probes = len(eps_sum)
    fig = plt.figure(figsize = (4*probes,4/2*N_atoms+2))
    gs = gridspec.GridSpec(N_atoms+2,2*probes)

    
    learn_ax = fig.add_subplot(gs[0:2,:])
    learn_ax.set_xlim([0,len(Elist_list[0])])

    learn_ax.set_xlabel('Episode')
    learn_ax.set_ylabel('Energy')
    learn_ax.plot(np.mean(np.array(Elist_list),axis=0),color='k',alpha=1,linewidth=4)
    for Elist in Elist_list:
        learn_ax.plot(Elist,color='k',alpha=0.02,linewidth=2)
    
    
    ax_list = []
    for i in range(N_atoms):
        for j in range(2*probes):
            ax_list.append(fig.add_subplot(gs[i+2,j]))
        
    ax = np.array(ax_list)# .reshape((N_atoms*probes,2)).T

    # ax = fig.subplots(N_atoms,2*probes)
    # ax= ax.T

    # stop
    
    for i,summary in enumerate(eps_sum):
        molecule_axes = []
        feature_axes = []        
        
        for n in range(N_atoms*probes*2):
            if np.mod(n,2*probes)==2*i:
                molecule_axes.append(ax[n])
            if np.mod(n,2*probes)==2*i+1:
                feature_axes.append(ax[n])

        


        mb_list = summary[0]
        fb_list = summary[1]
        sb_list = summary[2]
        pb_list = summary[3]

        # Plot molecules and probabilities
        for j,axis in enumerate(molecule_axes):
            axis.set_xlim([-5,5])
            axis.set_ylim([-5,5])
            

            if j<N_atoms-1:
                sp = np.array([LJEnv.gridToXY(point) for point in all_points[sb_list[j]]])
                # Create colors from probs list
                c = []
                max_p = np.max(pb_list[j])

        
                for p in pb_list[j]:
                    col = 1*(1-p)
                    c.append([col,col,col])
                axis.scatter(sp.T[0],sp.T[1],c=np.array(c).reshape((len(sp),3)))
        
            xylist = np.array([LJEnv.gridToXY(grid) for grid in mb_list[j]])
            axis.scatter(xylist.T[0],xylist.T[1])

        max_feat = np.max(fb_list)
    
        for j,axis in enumerate(feature_axes):
            axis.set_ylim([0,max_feat*1.1])
            axis.plot(fb_list[j],'r')



    fig.tight_layout()
    fig.savefig(name,dpi=200)
    

    

    

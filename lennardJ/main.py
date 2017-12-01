import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from doubleLJ import *
from fingerprintFeature import *
from plot_structure_and_feature import *

params = np.array([1,1,1])

N = 4

xlist1 = np.array([1,2,3,4])
ylist1 = np.array([1,1,1,1])

X1 = np.zeros(2*N)
for i in range(N):
    X1[2*i] = xlist1[i]
    X1[2*i+1] = ylist1[i]    
    

fpf = fingerprintFeature()
feat1 = fpf.get_singleFeature(X1)




xlist2 = np.array([0,0,1/2,1])
ylist2 = np.array([0,1,1/2,0])

X2 = np.zeros(2*N)
for i in range(N):
    X2[2*i] = xlist2[i]
    X2[2*i+1] = ylist2[i]

feat2 = fpf.get_singleFeature(X2)







# plt.plot(a)

fig = plt.figure()
ax_struct1 = fig.add_subplot(221)
ax_struct1.plot(xlist1,ylist1,'ro')

ax_feat1 = fig.add_subplot(222)
ax_feat1.plot(feat1,'k')



ax_struct2 = fig.add_subplot(223)
ax_struct2.plot(xlist2,ylist2,'bo')

ax_feat2 = fig.add_subplot(224)
ax_feat2.plot(feat2,'k')












fig.tight_layout()
fig.savefig('structFeatFig',dpi=400)


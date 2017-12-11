import numpy as np
from matplotlib import pyplot as plt
from LJEnvironment import *


r0 = 1.0
eps = 0
sigma= 1
params = (r0,eps,sigma)


LJEnv = LJEnvironment(params)


gridlist = np.load('best_grid.npy')

gridlist[-1][0] = 2
gridlist[-1][1] = -5

gridlist[-2][0] = 4
gridlist[-2][1] = -5

gridlist[-5][0] = -2
gridlist[-5][1] = 0


E = LJEnv.getEnergy(np.array([LJEnv.gridToXY(grid) for grid in gridlist]))

xylist = np.array([LJEnv.gridToXY(grid) for grid in gridlist])

fig = plt.figure()
ax = fig.gca()
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.plot(xylist.T[0],xylist.T[1],'bo')

fig.savefig('gridPlot_31_atoms_TEST')

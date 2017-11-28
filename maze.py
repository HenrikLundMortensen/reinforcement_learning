import numpy as np
from matplotlib import pyplot as plt

class Maze():
    """
    """
    
    def __init__(self, height, width):
        """
        
        Arguments:
        - `height`:
        """
        self._height = height
        self._width = width
        self._build_maze()

        
        self.start_state = [1,1]

        self.state = np.array(self.start_state)
        # self.target = np.array([self._width-2,self._height-2]) 
        self.target = np.array([self._width-2,1])       
        self.memory = []
        self.border_penalty = -5
        self.step_penalty = -0.1
        self.target_reward = 5
        self.score = 0


    def _build_maze(self):
        """
        
        Arguments:
        - `self`:
        """

        h = self._height
        w = self._width

        self.border = []
        self.maze_grid = np.zeros(shape=(w,h),dtype=object)
        for y in range(h):
            for x in range(w):
                self.maze_grid[x,y] = [x,y]

                if x == w-1 or y == h-1 or x == 0:
                    self.border.append([x,y])

                if y == 0 and x > 0 and x < w:
                    self.border.append([x,y])

                if y == 1 and x == int(self._width/2):
                    self.border.append([x,y])


    def restart(self):
        """
        
        Arguments:
        - `self`:
        """
        self.state = np.array(self.start_state)
        self.memory = []
                    
    def take_action(self,action):
        """
        
        Arguments:
        - `action`: 0 (up), 1 (down), 2 (right), 3 (left)
        """

        if action == 0:
            self.state[1] += 1

        if action == 1:
            self.state[1] += -1

        if action == 2:
            self.state[0] += 1

        if action == 3:
            self.state[0] += -1

        self.memory.append(self.state.tolist())
            
    def get_positional_reward(self):
        """
        
        Arguments:
        - `self`:
        """
        reward = 0
        termination = False
        if self.state.tolist() in self.border:
            reward += self.border_penalty
            termination = True

        if self.state[0]==self.target[0] and self.state[1]==self.target[1]:
            reward += self.target_reward
            termination = True
            
        return reward,termination 
        
                    

    def draw_maze(self,ax):
        """
        
        Arguments:
        - `self`:
        """

        ax.cla()
        
        for xy in self.border:
            ax.plot(xy[0]+0.5,xy[1]+0.5,marker='x',markersize=5,color='k')

        for state in np.array(self.memory):
            ax.plot(state[0]+0.5,state[1]+0.5,marker='.',markersize=4,color='r')
            
        ax.plot(self.state[0]+0.5,self.state[1]+0.5,color='r',marker='o',markersize=10)
        ax.plot(self.target[0]+0.5,self.target[1]+0.5,color='b',marker='d',markersize=10)


        
        # plt.pause(0.01)

        

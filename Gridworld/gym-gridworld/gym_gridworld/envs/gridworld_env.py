import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

'''
Desc:
12 x 12 gridworld.

Actions:
integers   actions
0          north 
1          east 
2          south 
3          west 

Observations:
Num     Observation     min     max
0       x-coordinate    0       11
1       y-coordinate    0       11

Reward:
0 for each step
10 on reaching goal
-1 (or) -2 (or) -3 (depending on which puddle), on entering a puddle

Start state:
4 positions assigned with equal probability

Terminal state:
On reaching goal, or
on completion of 500 time steps.

'''

class GridWorld(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.observation_space = spaces.Box(low = np.array([0, 0]), high = np.array([11, 11]), dtype = np.int32)
        self.action_space = spaces.Discrete(4)
        self.done = 0
        self.rmatrix = np.array([[0 for y in range(12)] for x in range(12)])
        
        # reward matrix
        a = 3
        b = 8
        r = -1
        for i in range(3):
            for j in range(a, b):
                self.rmatrix[11-a][j] = r
                self.rmatrix[11-b-1][j] = r
                self.rmatrix[11-j-1][a] = r
                self.rmatrix[11-j-2][b] = r
                self.rmatrix[11-a-1][b-1] = r
                self.rmatrix[11-a-2][b-1] = r
                
            a += 1
            b -= 1
            r -= 1
        
    # taking action
    def step(self, action):
        reward = 0
        p = np.random.rand()
        if action == 0:
            if p < 0.9:
                self.x -= 1
            elif 0.9 <= p < (0.9 + 0.1/3):
                self.y += 1
            elif (0.9 + 0.1/3) <= p < (0.9 + 0.2/3):
                self.y -= 1
            else:
                self.x += 1

        if action == 1:
            if p < 0.9:
                self.y += 1
            elif 0.9 <= p < (0.9 + 0.1/3):
                self.x += 1
            elif (0.9 + 0.1/3) <= p < (0.9 + 0.2/3):
                self.x -= 1
            else:
                self.y -= 1

        if action == 2:
            if p < 0.9:
                self.x += 1
            elif 0.9 <= p < (0.9 + 0.1/3):
                self.x -= 1
            elif (0.9 + 0.1/3) <= p < (0.9 + 0.2/3):
                self.y += 1
            else:
                self.y -= 1

        if action == 3:
            if p < 0.9:
                self.y -= 1
            elif 0.9 <= p < (0.9 + 0.1/3):
                self.y += 1
            elif (0.9 + 0.1/3) <= p < (0.9 + 0.2/3):
                self.x += 1
            else:
                self.x -= 1

        '''
        # Westerly wind
        if np.random.rand() <= 0.5:
            self.y += 1
        '''

        # if actions take agent outside the grid
        if self.x > 11:
            self.x = 11
        if self.x < 0:
            self.x = 0
        if self.y > 11:
            self.y = 11
        if self.y < 0:
            self.y = 0

        reward = self.rmatrix[self.x][self.y]

        #A
        '''
        if ((self.x == 0) and (self.y == 11)):
            reward += 10
            self.done = 1
        '''


        #B
        '''
        if ((self.x == 2) and (self.y == 9)):
            reward += 10
            self.done = 1
        '''

        #C
        
        if ((self.x == 6) and (self.y == 7)):
            reward += 10
            self.done = 1
        

        self.state = np.array([self.x, self.y])
        return self.state, reward, self.done, {}



    # resretting the env
    def reset(self):
        self.y = 0
        self.x = np.random.choice([5, 6, 10, 11])
        self.done = 0
        self.state = np.array([self.x, self.y])
        return self.state

    def render(self, mode = 'human'):        
        return
                  
        




        
        

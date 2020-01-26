import gym
import gym_gridworld
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics


# Initializations
gamma = 0.9
alpha = 0.05
epsmin = 0.01
decay_rate = 0.99
episodes = 2000
exp = 50
avg_t = [[0 for i in range(episodes)] for j in range(exp)]
avg_r = [[0 for i in range(episodes)] for j in range(exp)]
time = [0 for i in range(episodes)]
reward = [0 for i in range(episodes)]


# epsilon greedy
def epsgreedy(x, y, Q):
        global epsilon
        if np.random.rand() < epsilon:
            return random.randrange(env.action_space.n)
        return np.random.choice([ind for ind in range(env.action_space.n) if Q[x][y][ind] == np.amax(Q[x][y])])


env = gym.make('gridworld-v0')


for e in range(exp):
    print(e)

    # Initialize table for Q values 
    Q = [[[0 for k in range(env.action_space.n)] for j in range(12)] for i in range(12)] 
    epsilon = 1

    for ep in range(episodes):
        tot_r = 0
        s = env.reset()
        for t in range(500):
            env.render()
            x, y = s
            a = epsgreedy(x, y, Q)
            s1, r, done, info = env.step(a)
            x1, y1 = s1

            #Q-learning update
            Q[x][y][a] = Q[x][y][a] + alpha*(r + gamma*np.amax(Q[x1][y1]) - Q[x][y][a])
            s = s1
            tot_r += r
            if epsmin < epsilon:
                epsilon *= decay_rate 
            

            if done:
                print(ep)
                print('solved! :)')
                avg_t[e][ep] = t
                avg_r[e][ep] = tot_r
                print(t)
                print(epsilon)
                print(tot_r)    
                print('\n')
                break

        if not done:
            print('couldn\'t be solved :(')


# creating lists for plotting
for i in range(episodes):
    for j in range(exp):
        time[i] += avg_t[j][i]
        reward[i] += avg_r[j][i] 
    time[i] /= exp
    reward[i] /= exp


# displaying optimal policies obtained
a = [[[ind for ind in range(env.action_space.n) if Q[x][y][ind] == np.amax(Q[x][y])] for y in range(12)] for x in range(12)]
for x in range(12):
    print(a[x])


# plotting
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('QLearn: Average over last %d independent runs' % exp)
ax1.plot([ep for ep in range(episodes)], [i for i in time])
ax2.plot([ep for ep in range(episodes)], [i for i in reward])
ax1.set_xlabel('No. of episodes')
ax2.set_xlabel('No. of episodes')
ax1.set_ylabel('No. of steps to goal')
ax2.set_ylabel('Reward per episode')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




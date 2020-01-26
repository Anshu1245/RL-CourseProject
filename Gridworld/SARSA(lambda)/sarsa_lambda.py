import gym
import gym_gridworld
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics


# Initialization
gamma = 0.9
alpha = 0.05
epsmin = 0.01
decay_rate = 0.995
episodes = 2000
exp = 50


# epsilon greedy
def epsgreedy(x, y, Q):
        global epsilon
        if np.random.rand() < epsilon:
            return random.randrange(env.action_space.n)
        return np.random.choice([ind for ind in range(env.action_space.n) if Q[x][y][ind] == np.amax(Q[x][y])])

env = gym.make('gridworld-v0')


# Initializing plot
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlabel('No. of episodes')
ax2.set_xlabel('No. of episodes')
ax1.set_ylabel('No. of steps to goal')
ax2.set_ylabel('Reward per episode')


for lamb in [0, 0.3, 0.5, 0.9, 0.99, 1.0]:
    avg_t = [[0 for i in range(episodes)] for j in range(exp)]
    avg_r = [[0 for i in range(episodes)] for j in range(exp)]
    time = [0 for i in range(episodes)]
    reward = [0 for i in range(episodes)]

    for e in range(exp):
        print(e)

        Q = [[[0 for k in range(env.action_space.n)] for j in range(12)] for i in range(12)]     
        epsilon = 1

        for ep in range(episodes):

            # eligibility trace
            el_trace = [[[0 for k in range(env.action_space.n)] for j in range(12)] for i in range(12)]
            tot_r = 0
            s = env.reset()
            x, y = s
            a = epsgreedy(x, y, Q)
            for t in range(500):
                for k in range(env.action_space.n):
                    el_trace[x][y][k] = 0
                el_trace[x][y][a] = 1
                for i in range(12):
                    for j in range(12):
                        for k in range(env.action_space.n):
                            if (i != x) or (j != y):
                                el_trace[i][j][k] *= (gamma*lamb)

                env.render()
                s1, r, done, info = env.step(a)
                x1, y1 = s1
                a1 = epsgreedy(x1, y1, Q)

                # SARSA(lambda) update
                for i in range(12):
                    for j in range(12):
                        for k in range(env.action_space.n):
                            Q[i][j][k] = Q[i][j][k] + alpha*(r + gamma*(Q[x1][y1][a1]) - Q[x][y][a])*el_trace[i][j][k]
                x, y = s1
                a = a1
                tot_r += r
                if epsmin < epsilon:
                    epsilon *= decay_rate 
                

                if done:
                    print(ep)
                    print('solved! :)')
                    if ep >= 25:
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
    for i in range(25, episodes):
        for j in range(exp):
            time[i] += avg_t[j][i]
            reward[i] += avg_r[j][i] 
        time[i] /= exp
        reward[i] /= exp


    ax1.plot([ep for ep in range(25, episodes)], [i for i in time[25:]], label = '%0.2f' % lamb)
    ax2.plot([ep for ep in range(25, episodes)], [i for i in reward[25:]], label = '%0.2f' % lamb)


# plotting
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'lower right')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('SARSA-Lambda (For task C): Average over last %d independent runs' % exp)
plt.show()





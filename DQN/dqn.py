import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import statistics
from keras import initializers



# Initializations
epsilon = 1
gamma = 0.95                                     # The "gamma" of the problem is assumed to be 0.95
memory = deque(maxlen = 1000)
batch = 64
score = []
count = 0
avg_tot_r = []
maxavg = 0
counter = 0



# making the environment
env = gym.make('CartPole-v0')



# setting the seeds for every randomness possible
# only except the randomized initialization of the weights of the neural network
env.seed(0)
np.random.seed(0)
random.seed(100)



# the neural net for learning the action value function
Q = Sequential()
Q.add(Dense(24, input_shape = env.observation_space.shape, activation = 'relu'))
Q.add(Dense(24, activation = 'relu'))
Q.add(Dense(env.action_space.n, activation = 'linear'))
Q.compile(loss = 'mse', optimizer = adam(lr = 0.001))

# the neural net to be #The "gamma" of the problem is assumed to be 0.95used as the target network
Qt = Sequential()
Qt.add(Dense(24, input_shape = env.observation_space.shape, activation = 'relu'))
Qt.add(Dense(24, activation = 'relu'))
Qt.add(Dense(env.action_space.n, activation = 'linear'))
Qt.compile(loss = 'mse', optimizer = adam(lr = 0.001))

# resetting weights of the the target network to that of action-value netwrok 
Qt.set_weights(Q.get_weights())


# function for epsilon-greedy action selection
def epsgreedy(state):
    if np.random.rand() <= epsilon:
        return random.randrange(env.action_space.n)
    q = Q.predict(state)
    return np.argmax(q)


# function for experience replay
def exp_replay():
    if len(memory) < batch:
        return

    global epsilon
    minibatch = random.sample(memory, batch)
    states = np.array([i[0] for i in minibatch])
    actions = np.array([i[1] for i in minibatch])
    rewards = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])
    dones = np.array([i[4] for i in minibatch])

    # Removing the reduntant one dimensions
    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    targets = rewards + gamma*(np.amax(Qt.predict_on_batch(next_states), axis=1))*(1-dones)
    # targets = rewards + gamma*(np.amax(Q.predict_on_batch(next_states), axis=1))*(1-dones)
    targets1 = Qt.predict_on_batch(states)
    # targets1 = Q.predict_on_batch(states)
    indice = np.array([i for i in range(batch)])
    targets1[[indice], [actions]] = targets      # to make sure the values are updated only for the sampled actions                

    # updation of the action-value network
    Q.fit(states, targets1, epochs = 1, verbose = 0)
    
    # epsilon decay
    if epsilon>0.01:
        epsilon *= 0.995                         


for ep in range(1, 1000):
    tot_r = 0
    s = env.reset()
    s = np.reshape(s, (1, 4))
    for t in range(1000):
        env.render()
        a = epsgreedy(s)
        s1, r, done, x = env.step(a)
        tot_r += r
        s1 = np.reshape(s1, (1, 4))
        memory.append((s, a, r, s1, done))       #committing to replay memory
        s = s1
        exp_replay()
        count += 1
        if count == 100:
            Qt.set_weights(Q.get_weights())      # resetting weights of target network to that of action-value network
            count = 0

        if done:
            print(ep)                
            print(tot_r)
            score.append(tot_r)
            if ep >= 100:
                print(statistics.mean(score[ep-100:ep]))                
                avg_tot_r.append(statistics.mean(score[ep-100:ep]))
            print('\n')
            break

    # checking whether environment is solved
    if (statistics.mean(score[ep-100:ep]) >= 195) and (ep >= 100):
        print("SOLVED!")
        counter = 1
        break

    
if counter == 0:
    print('Couldn\'t be solved')

# Plotting
else:
    plt.plot([epsiode for epsiode in range(100, ep+1)], [i for i in avg_tot_r])
    plt.title('SOLVED CartPole-v0 in %d episodes!' % ep)
    plt.xlabel('Number of episodes')
    plt.ylabel('Average over the last 100 episodes')
    plt.show()






    



    

    


    





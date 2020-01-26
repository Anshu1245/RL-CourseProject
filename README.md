# RL-CourseProject

## DQN
Reading and coding up the DQN paper to solve any of the discrete-space, classic control environments by OpenAI gym. Here *'CartPole-v0'* is used. Testing the proposed measures, namely the target network and the replay memory for a general case of function approximation using a feed-forward neural net and observing the learning curves (average reward over the last 100 episodes vs the number of episodes). <br/>
Environment considered solved if/when the curve touches 195.

Reporting plots on performances, observations and inferences on hyperparameter variation. 

Analysis of performance and inferences drawn on removal of the target network and the transition replay buffer (as proposed by the original DQN paper for smooth and stable learning of the neural nets). Conclusion drawn on the relative importance of either based on the breakdown of learning observed from the plots.  

*Reference*: DQN paper (https://arxiv.org/pdf/1312.5602)

## Gridworld

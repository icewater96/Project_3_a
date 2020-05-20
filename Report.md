# Report

## Learning algorithms

This implementation uses Deep Reinforcement Learning approach. To be more specific, Multi-Agent Deep Deterministic Policy Gradient (MADDPG) is utilized to train an actor and critic network. The training process involves Experience Replay to break unwanted correlation in experiences. Target networks for the actor and critic are also implemented to increase stability of training. The target networks are updates every UPDATE_EVERY times by soft update method. 

The actor network consist of 3 dense layers and the critic network consists of 3 dense layers as well. 

Other hyperparameters are listed below:
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 3e-4        # learning rate of the critic
- WEIGHT_DECAY_actor = 0.0 # L2 weight decay
- WEIGHT_DECAY_critic = 0.0 # L2 weight decay
- NOISE_START=1.0
- NOISE_END=0.1
- NOISE_REDUCTION=0.999
- EPISODES_BEFORE_TRAINING = 300
- NUM_LEARN_STEPS_PER_ENV_STEP = 3

## Trainig 
The average score over 100 consecutive episodes reached 30.0 at 824th Episode as following: 

>Episode 1610, Average Score: 0.3547, Current Score:0.1000

>Episode 1620, Average Score: 0.3588, Current Score:0.5000

>Episode 1630, Average Score: 0.4068, Current Score:0.6000

>Episode 1640, Average Score: 0.3719, Current Score:0.2000

>Episode 1650, Average Score: 0.4289, Current Score:0.1000

>Episode 1660, Average Score: 0.4559, Current Score:0.8000

>Episode 1670, Average Score: 0.4869, Current Score:0.3000

>Episode 1680, Average Score: 0.5039, Current Score:0.1000

>Episode 1690, Average Score: 0.5819, Current Score:0.2000

>Episode 1700, Average Score: 0.6259, Current Score:0.1000

>Episode 1710, Average Score: 0.6610, Current Score:0.1000

>Episode 1720, Average Score: 0.7380, Current Score:2.5000

>Episode 1730, Average Score: 0.7200, Current Score:0.1000

>Episode 1740, Average Score: 0.8120, Current Score:0.3000

>Episode 1750, Average Score: 0.8350, Current Score:0.9000

>Episode 1760, Average Score: 0.8550, Current Score:0.2000

>Episode 1770, Average Score: 0.8929, Current Score:0.1000

>Episode 1780, Average Score: 0.9179, Current Score:0.1000

>Episode 1790, Average Score: 0.8739, Current Score:2.4000

>Episode 1800, Average Score: 0.9016, Current Score:2.6000

>Episode 1810, Average Score: 0.9786, Current Score:1.0000

>Episode 1820, Average Score: 0.9386, Current Score:0.0000

>Episode 1830, Average Score: 0.9624, Current Score:0.3000

## Plot of rewards
This plot demonstrates the training process. The Y-Axis is average of reward over 100 episodes. As the training goes on, the average reward over 100 episode reached above 30.0. 

![](Figure_2020-05-08_001519.png)


## Future work
This implement works fine but it takes a long time to train. I would like to try to optimize hyper-parameters to improve performance. 

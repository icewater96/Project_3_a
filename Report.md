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

> Episode 820	Average Score: 29.75	

> Episode 821	Average Score: 29.80	

> Episode 822	Average Score: 29.86	

> Episode 823	Average Score: 29.93	

> Episode 824	Average Score: 30.00	

> Episode 825	Average Score: 29.99	

> Episode 826	Average Score: 30.06	

> Episode 827	Average Score: 30.15	

> Episode 828	Average Score: 30.31	

> Episode 829	Average Score: 30.31	

## Plot of rewards
This plot demonstrates the training process. The Y-Axis is average of reward over 100 episodes. As the training goes on, the average reward over 100 episode reached above 30.0. 

![](Figure_2020-05-08_001519.png)


## Future work
This implement works fine but it takes a long time to train. I would like to try to optimize hyper-parameters to improve performance. 

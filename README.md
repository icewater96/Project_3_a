# Project_3

This repository is Project 3 Collaboration and Competition in Udacity Deep Reinforcement Learning NanoDegree. 

## Getting started

The framework is based on Pytorch and Python 3.6. 

My implementation is based on Windows 10 OS and did the following from the dependencies section of https://github.com/udacity/deep-reinforcement-learning:

conda create --name drlnd python=3.6

conda activate drlnd

pip install gym

pip install unityagents==0.4.0

pip install .

conda install pytorch=0.4.0 -c pytorch  # Some errors occurred when I ran the above line. My Undacity mentor suggested this line. 

Once the above steps are done, one needs to clone this repository to get the working code and saved model weights.

## Project environment

This environment simulates two players/agents bouncing a ball over net by controlling rackets. If an agent succeeds in hitting the ball over the net, it will receive a reward of 0.1. On the other side, if an agent lets a ball hit the ground or go out of bounds, it will receive a punishing reward of -0.01. The goal of training is to keep the ball in play as long as possible. The environment is episodic. The score of each episode is the maximum of accumulated rewards of the two agents without discounting. When the average of score over 100 episodes reaches 0.5 the task is considered solved. 

Each agent can move in horizontally and vertically, making a 2-dimensionaly continuouse action space. The obseration space consists of 8 variables representing the positions and velocities of the ball and racket. 

## How to run

Python code main_project_3.py and lib.py in the repository contains all necessary code for this project. Running main_project_3.py will start the training process automatically. The script will save intermediate models into checkpoint file. 

A set of pretrained models have been already saved in this repository as .pth files. 


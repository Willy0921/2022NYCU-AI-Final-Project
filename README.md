# 2022NYCU-AI-Final-Project
## Overview
  Atari-freeway is a game that a player can control a chicken to get to the other side of a ten-lane highway with heavy traffic flow. Once hit by a car, the chicken will be forced back slightly. While dodging the vehicles, we have to make it reach to the endpoint as fast as it can since our goal is to  make the chickens that cross the road safely as many as possible within a limited time. That is, we must to train our chicken to march in an efficient way.  
Among the algorithms in reinforcement learning, we chose Q-learning and DQN to train our agent. Thus we can improve it by using DQN to approximate Q-function by neural network.  
![image](https://github.com/Willy0921/2022NYCU-AI-Final-Project/blob/main/Freeway%20AI/freeway.png)

## Environment
Run the following command to set the environment of Atari Freeway  
`pip install gym[atari]`  


## Parameters
Run the following command to modify and check the imformation of parameters  
`python main.py -h`

from ale_py import ALEInterface
from ale_py.roms import Freeway
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gym
import random

seed = 199
SEED = 199
env = gym.make('Freeway-v4',  obs_type='ram')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(SEED)
env.action_space.seed(SEED)
env.reset()
x = 0
y = 0
step = 0
dic = {}
while True:
    #action = env.action_space.sample()
    action = 1
    next_state, reward, done, _ = env.step(action)  # take a random action
    x = next_state.tostring()
    dic[x] = 1
    print(dic[x])
    step += 1
    '''if reward:
        print(reward)
        x += 1
        print("reward: ", x)
    if done:
        y += 1
        print("step: ", step)
        break'''
    #print(x, "   ", y)
#print("reward: ", x, "   ", y)
env.close()

from ale_py import ALEInterface
from ale_py.roms import Freeway

import gym
env = gym.make('Freeway-v0',  render_mode='human')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())  # take a random action
env.close()

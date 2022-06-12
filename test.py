import argparse
import torch
import numpy as np
import os
import gym
from DQN_train import Agent
from ale_py import ALEInterface
from ale_py.roms import Freeway

parser = argparse.ArgumentParser()
parser.add_argument("--test_times", type=int, default=100)
parser.add_argument("--path", type=str, default="./Test_modals/")
parser.add_argument("--algorithm", type=str, default="DQN/")
parser.add_argument("--file", type=str,
                    default="DQN_traintimesXepisode_1x300.pt")
args = parser.parse_args()


def test(env):
    env.reset()
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(
        torch.load(args.path + args.algorithm + args.file))
    for i in range(args.test_times):
        print(f"#{i + 1} testing progress")
        state = env.reset()
        count = 0
        while True:
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step(action)
            count += reward
            if done:
                rewards.append(count)
                print(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    print(f"max :{testing_agent.check_max_Q()}")


if __name__ == "__main__":

    env = gym.make('Freeway-v4',  obs_type='ram')
    test(env)

env.close()

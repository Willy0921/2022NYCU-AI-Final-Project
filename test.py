import argparse
import torch
import numpy as np
import os
import gym
from DQN_train import Agent
from ale_py import ALEInterface
from ale_py.roms import Freeway

parser = argparse.ArgumentParser()
parser.add_argument("--test_times", type=int, default=10)
parser.add_argument("--algorithm", type=str, default="DQN")
parser.add_argument("--compare", type=str, default="traintimesXepisode")
parser.add_argument("--file", type=str, default="DQN_traintimesXepisode_3x100")
args = parser.parse_args()


def test(env):

    read_file = args.file + ".pt"
    output_file = args.file + ".txt"
    read_path = "./Train_data/" + args.algorithm + "/tables/" + read_file
    output_path = "./Test_results/" + args.algorithm + \
        "/" + args.compare + "/" + output_file

    if not os.path.exists("./Test_results/" + args.algorithm +
                          "/" + args.compare):

        os.mkdir("./Test_results/" + args.algorithm +
                 "/" + args.compare)

    f = open(output_path, 'w')
    f.write(args.file + '\n')
    env.reset()
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(
        torch.load(read_path))

    for i in range(args.test_times):
        state = env.reset()
        score = 0
        while True:
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                rewards.append(score)
                f.write(str(score) + ' ')
                print(f"#{i + 1} testing progress   score: {score}")
                break
            state = next_state
    avg_reward = np.mean(rewards)
    f.write('\n')
    f.write(str(avg_reward))
    print("average reward: ", avg_reward)
    f.close()


if __name__ == "__main__":

    env = gym.make('Freeway-v4',  obs_type='ram')
    test(env)

env.close()

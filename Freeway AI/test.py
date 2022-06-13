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
parser.add_argument("--algorithm", type=str,
                    default="Qlearning", help="Determines test algorithm")
parser.add_argument("--compare", type=str, default="num_bins")
parser.add_argument("--file", type=str, default="Qlearning_num_bins_8",
                    help="Name of table that will be test")
parser.add_argument("--num_bins", type=int, default=2)
args = parser.parse_args()


def DQN_test(env):

    read_file = args.file + ".pt"
    output_file = args.file + ".txt"
    read_path = "./Train_data/" + args.algorithm + "/tables/" + read_file
    output_path = "./Test_results/" + args.algorithm + \
        "/" + args.compare + "/" + output_file

    if not os.path.exists("./Test_results/" + args.algorithm + "/" + args.compare):
        os.mkdir("./Test_results/" + args.algorithm + "/" + args.compare)

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
    print(f"average reward: {avg_reward}")
    f.close()


def Q_learning_test(env):
    """
    Test the agent on the given environment.
    """

    def init_bins(lower_bound, upper_bound, num_bins):
        size = (upper_bound - lower_bound)/num_bins
        return np.arange(lower_bound, upper_bound, size)[1:]

    bins = init_bins(0, 256, args.num_bins)

    def discretize_value(value, bins):
        return np.searchsorted(bins, value, side="right")

    def discretize_observation(observation):
        answer = []
        for i in range(len(observation)):
            answer.append(discretize_value(observation[i], bins))
        return np.array(answer)

    read_file = args.file + ".npy"
    output_file = args.file + ".txt"
    read_path = "./Train_data/" + args.algorithm + "/tables/" + read_file
    output_path = "./Test_results/" + args.algorithm + \
        "/" + args.compare + "/" + output_file

    if not os.path.exists("./Test_results/" + args.algorithm + "/" + args.compare):
        os.mkdir("./Test_results/" + args.algorithm + "/" + args.compare)

    env.reset()
    testing_agent = Agent(env)

    testing_agent.qtable = np.load(read_path, allow_pickle=True).item()
    rewards = []

    f = open(output_path, 'w')
    f.write(args.file + '\n')

    for i in range(args.test_times):
        state = discretize_observation(testing_agent.env.reset())
        score = 0
        while True:
            state = np.array2string(
                state, max_line_width=130, separator='')[1:-1]
            action = np.argmax(testing_agent.qtable.setdefault(
                state, [0.] * testing_agent.env.action_space.n))
            next_observation, reward, done, _ = testing_agent.env.step(action)
            if reward:
                score += 1
            next_state = discretize_observation(next_observation)

            if done == True:
                f.write(str(score) + ' ')
                print(f"#{i + 1} testing progress   score: {score}")
                rewards.append(score)
                break

            state = next_state

    avg_reward = np.mean(rewards)
    f.write('\n')
    f.write(str(avg_reward))

    print(f"average reward: {avg_reward}")


if __name__ == "__main__":

    # env = gym.make('Freeway-v4',  obs_type='ram', render_mode='human')
    env = gym.make('Freeway-v4',  obs_type='ram')

    if args.algorithm == "DQN":
        DQN_test(env)
    elif args.algorithm == "Qlearning":
        Q_learning_test(env)

    env.close()

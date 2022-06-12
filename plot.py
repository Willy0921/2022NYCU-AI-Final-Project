import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('Freeway-v4')
    plt.xlabel('epsisode')
    plt.ylabel('score')


def DQN():

    DQN_Rewards = np.load(
        "./Rewards/DQN_traintimesXepisode_2x150.npy").transpose()
    # DQN_Rewards = np.load("./Rewards/DQN_rewards_ratio_1000.npy")
    # print(DQN_Rewards)
    # print("shape: ", DQN_Rewards.shape)
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot()

    plt.plot([i for i in range(150)], DQN_avg,
             label='DQN', color='blue')
    plt.fill_between([i for i in range(150)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN_traintimesXepisode_2x150.png")
    plt.show()
    plt.close()


def compare():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy").transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    Q_learning_Rewards = np.load("./Rewards/cartpole_rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    initialize_plot()
    plt.plot([i for i in range(1000)], DQN_avg, label='DQN', color='blue')
    plt.plot([i for i in range(1000)],
             Q_learning_avg[:1000], label='Q_learning', color='orange')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DQN", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("./Graphs"):
        os.mkdir("./Graphs")

    if args.DQN:
        DQN()
    elif args.compare:
        compare()

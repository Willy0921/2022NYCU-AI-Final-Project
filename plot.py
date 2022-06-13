import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--DQN", action="store_true")
parser.add_argument("--multi", action="store_true")
parser.add_argument("--file1", type=str,
                    default="DQN_episode_1000")
parser.add_argument("--compare", type=str, default="episode")
parser.add_argument("--algorithm", type=str, default="DQN")
args = parser.parse_args()


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('Freeway-v4')
    plt.xlabel('epsisode')
    plt.ylabel('score')


def DQN():

    read_path = "./Train_data/" + args.algorithm + "/rewards/" + args.file1 + ".npy"
    output_path = "./Graphs/" + args.algorithm + "/" + args.file1 + ".png"

    DQN_Rewards = np.load(read_path).transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot()

    plt.plot([i for i in range(len(DQN_Rewards))],
             DQN_avg, label='DQN', color='blue')
    plt.fill_between([i for i in range(len(DQN_Rewards))], DQN_avg +
                     DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def multi():

    output_path = "./Graphs/" + args.algorithm + "/" + args.compare + ".png"

    rewards = []
    labels = []
    path = "./Train_data/" + args.algorithm + "/rewards/"
    for filename in os.listdir(path):
        if args.algorithm + "_" + args.compare in filename:
            rewards.append(np.load(path + filename).transpose())
            labels.append(filename[0:-4])

    avgs = []
    for reward in rewards:
        avgs.append(np.mean(reward, axis=1))

    initialize_plot()

    colors = ['green', 'red', 'blue', 'orange']

    for i in range(len(avgs)):
        plt.plot([j for j in range(len(avgs[i]))],
                 avgs[i], label=labels[i], color=colors[i])

    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.show()
    plt.close()


if __name__ == "__main__":

    if not os.path.exists("./Graphs/"):
        os.mkdir("./Graphs/")

    if args.DQN:
        DQN()
    elif args.multi:
        multi()

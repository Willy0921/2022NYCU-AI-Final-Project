import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--DQN", action="store_true")
parser.add_argument("--multi", action="store_true")
parser.add_argument("--file", type=str,
                    default="Qlearning_episode_600(200)")
parser.add_argument("--algorithm", type=str, default="Qlearning")
parser.add_argument("--compare", type=str, default="episode")
args = parser.parse_args()


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('Freeway-v4')
    plt.xlabel('epsisode')
    plt.ylabel('score')


def single():

    read_path = "./Train_data/" + args.algorithm + "/rewards/" + args.file + ".npy"
    output_path = "./Graphs/" + args.algorithm + "/" + args.file + ".png"

    rewards = np.load(read_path).transpose()
    avg = np.mean(rewards, axis=1)
    std = np.std(rewards, axis=1)

    initialize_plot()
    if args.algorithm == "DQN":
        plt.plot([i for i in range(len(rewards))],
                 avg, label='DQN', color='blue')
    else:
        plt.plot([i for i in range(len(rewards))],
                 avg, label='Q Learning', color='blue')
    plt.fill_between([i for i in range(len(rewards))],
                     avg + std, avg-std, facecolor='lightblue')
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

    if args.multi:
        multi()
    else:
        single()

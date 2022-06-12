import argparse
import torch
import numpy as np
import os
import gym
from DQN_train import Agent
from ale_py import ALEInterface
from ale_py.roms import Freeway

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="DQN")
parser.add_argument("--compare", type=str, default="traintimesXepisode")
args = parser.parse_args()


if __name__ == "__main__":

    dataset = []
    path = "./Test_results/" + args.algorithm + "/" + args.compare
    for filename in os.listdir(path):
        f = open(filename, 'r')

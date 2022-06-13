import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gym
import random
from tqdm import tqdm
from ale_py import ALEInterface
from ale_py.roms import Freeway
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="DQN_rewardsratio_1",
                    help="Determines the name of this modal")
parser.add_argument("--train_times", type=int, default=1,
                    help="Determines the times of training process")
parser.add_argument("--episode", type=int, default=250,
                    help="Determines the episode we want to train per time")

parser.add_argument("--epsilon", type=float, default=0.8,
                    help="Determines the explore/expliot rate of the agent")
parser.add_argument("--learning_rate", type=float, default=0.0002,
                    help="Determines the step size while moving toward a minimum of a loss function")
parser.add_argument("--GAMMA", type=float, default=0.97,
                    help="The discount factor (tradeoff between immediate rewards and future rewards")
parser.add_argument("--batch_size", type=int, default=32,
                    help="The number of samples which will be propagated through the neural network")
parser.add_argument("--capacity", type=int, default=10000,
                    help="The size of the replay buffer")

parser.add_argument("--inner_layer_size", type=int, default=256)
parser.add_argument("--hidden_layer_size", type=int, default=512)

parser.add_argument("--learn_threshold", type=int, default=10245)
parser.add_argument("--reward_ratio", type=int, default=1000)

args = parser.parse_args()

total_rewards = []
best_score = float('-inf')


class replay_buffer():
    '''
        - A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
            - Insert a sequence of data gotten by the agent into the replay buffer.
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
            - Sample a batch size of data from the replay buffer.
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self, num_actions, hidden_layer_size=args.hidden_layer_size):
        super(Net, self).__init__()
        self.input_state = 128  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(
            self.input_state, args.inner_layer_size)  # input layer
        self.fc2 = nn.Linear(args.inner_layer_size,
                             hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
           - Forward the state to the neural network.
           - Return a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=args.epsilon, learning_rate=args.learning_rate, GAMMA=args.GAMMA, batch_size=args.batch_size, capacity=args.capacity):
        '''
            - The agent learning how to control the action of the agent.
        '''
        self.env = env
        self.n_actions = 3  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
            - Implement the learning function.

            Steps:
            -----
            1. Update target net by current net every 100 times.
            2. Sample trajectories of batch size from the replay buffer.
            3. Forward the data to the evaluate net and the target net.
            4. Compute the loss with MSE.
            5. Zero-out the gradients.
            6. Backpropagation.
            7. Optimize the loss function.
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        b_memory = self.buffer.sample(self.batch_size)
        b_state = torch.FloatTensor(np.asarray(b_memory[0]))
        b_action = torch.LongTensor(np.asarray(
            b_memory[1])).view(self.batch_size, 1)
        b_reward = torch.IntTensor(np.asarray(
            b_memory[2])).view(self.batch_size, 1)
        b_next_state = torch.FloatTensor(np.asarray(b_memory[3]))
        b_done = torch.IntTensor(np.asarray(
            b_memory[4])).view(self.batch_size, 1)

        q_eval = self.evaluate_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * \
            q_next.max(1)[0].view(self.batch_size, 1)
        q_target = torch.mul(q_target, torch.logical_not(b_done))

        loss = nn.MSELoss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        '''
            - Implement the action-choosing function.
            - Choose the best action with given state and epsilon
        '''
        with torch.no_grad():
            x = torch.unsqueeze(torch.FloatTensor(state), 0)
            r = np.random.rand()
            if r < self.epsilon:
                p = random.uniform(0, 1)
                if (p < 0.3):
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = 1
            else:
                actions_value = self.evaluate_net(x)
                action = torch.max(actions_value, 1)[1].data.numpy()[0]
            return action


def train(env):
    '''
        - Trainning process: total 2049 steps in Freeway
    '''
    agent = Agent(env)
    rewards = []
    for _ in tqdm(range(args.episode)):
        state = env.reset()
        score = 0
        if agent.count >= args.learn_threshold:
            agent.epsilon = 0.05
        while True:
            agent.count += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.buffer.insert(state, int(action), reward *
                                args.reward_ratio, next_state, int(done))
            if agent.count >= args.learn_threshold:
                agent.learn()
            if done:
                rewards.append(score)
                break
            state = next_state

        global best_score
        if best_score <= score:
            best_score = score
            torch.save(agent.target_net.state_dict(), "./Train_data/DQN/tables/" +
                       args.file + "pt")

    total_rewards.append(rewards)


if __name__ == "__main__":

    env = gym.make('Freeway-v4',  obs_type='ram')

    env.reset()
    '''
        action_sapce:
            '0': stay in place
            '1': go forward
            '2': go backward
    '''
    if not os.path.exists("./Train_data/DQN/tables/"):
        os.mkdir("./Train_data/DQN/tables/")

    for i in range(args.train_times):
        print(f"#{i + 1} training process")
        train(env)
    print("best score in training process: ", best_score)

    if not os.path.exists("./Train_data/DQN/rewards/"):
        os.mkdir("./Train_data/DQN/rewards/")

    np.save("./Train_data/DQN/rewards/" +
            args.file + "npy", np.array(total_rewards))

    env.close()

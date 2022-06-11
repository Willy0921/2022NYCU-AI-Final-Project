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
parser.add_argument("--train_times", type=int, default=1)
parser.add_argument("--SEED", type=int, default=156)

parser.add_argument("--epsilon", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--GAMMA", type=float, default=0.97)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--capacity", type=int, default=10000)

parser.add_argument("--inner_layer_size", type=int, default=256)
parser.add_argument("--hidden_layer_size", type=int, default=512)

parser.add_argument("--episode", type=int, default=50)
parser.add_argument("--timesteps", type=int, default=500)
# total 2049 steps in Freeway
parser.add_argument("--learn_threshold", type=int, default=2500)

parser.add_argument("--test_times", type=int, default=1)
parser.add_argument("--reward_ratio", type=int, default=1)

args = parser.parse_args()

total_rewards = []
best_q_value = float('-inf')


class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish

        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network

        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
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
        Forward the state to the neural network.

        Parameter:
            states: a batch size of states

        Return:
            q_values: a batch size of q_values
        '''

        # print(states.shape)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=args.epsilon, learning_rate=args.learning_rate, GAMMA=args.GAMMA, batch_size=args.batch_size, capacity=args.capacity):
        """
        The agent learning how to control the action of the cart pole.

        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
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
        - Here are the hints to implement.

        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----

        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)

        Returns:
            None (Don't need to return anything)
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

        # b_state = b_state.view(self.batch_size, 100800)

        # print(b_state.shape)
        # print(b_action.shape)

        q_eval = self.evaluate_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * \
            q_next.max(1)[0].view(self.batch_size, 1)
        q_target = torch.mul(q_target, torch.logical_not(b_done))

        loss = nn.MSELoss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        global best_q_value
        q_value = torch.max(q_eval, 0)[0].data.numpy()[0]
        if best_q_value < q_value:
            best_q_value = q_value
            # print(q_value)
            torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon

        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)

        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # print(torch.FloatTensor(state).shape)
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
                print(actions_value)
                #print(torch.max(actions_value, 1)[1].data.numpy())
                action = torch.max(actions_value, 1)[1].data.numpy()[0]
                # print("action: ", action)
                # action = torch.max(action, 1)[1]
                # action = torch.max(action, 1)[1].data.numpy()[0]
                # print(action)
            return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state

        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)

        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        state = self.env.reset()
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        q_values = self.target_net(x)
        max_q = torch.max(q_values, 1)[0].data.numpy()[0]
        return max_q


def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
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
                rewards.append(reward)
                print(count)
                break
            state = next_state
    print(f"reward: {np.mean(count)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


def train(env):
    agent = Agent(env)
    rewards = []
    for _ in tqdm(range(args.episode)):
        state = env.reset()
        iteration_time = 0
        count = 0
        while iteration_time < args.timesteps:
            iteration_time += 1
            agent.count += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            count += reward
            # print(type(reward), " ", reward)
            # print(type(done), " ", done)
            if iteration_time == args.timesteps:
                done = 1
            agent.buffer.insert(state, int(action), reward *
                                args.reward_ratio, next_state, int(done))
            if agent.count >= args.learn_threshold:
                agent.epsilon = 0.05
                agent.learn()
            # if done:
                # rewards.append(reward)
            state = next_state
        rewards.append(count)
    total_rewards.append(rewards)


def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":

    env = gym.make('Freeway-v4',  obs_type='ram', render_mode='human')
    # env = gym.make('Freeway-v4')

    SEED = args.SEED
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    env.reset()
    '''
        action  '0': stay in place
                '1': go forward
                '2': go backward
    '''
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    for i in range(args.train_times):
        print(f"#{i + 1} training progress")
        train(env)

    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    '''for _ in range(1000):
        action = env.action_space.sample()
        # print(env.action_space)
        next_state, reward, done, _ = env.step(action)  # take a random action
        # print(next_state.shape)  # (210, 160, 3)
        print("-----------")'''

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()

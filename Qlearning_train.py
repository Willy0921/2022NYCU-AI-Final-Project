import numpy as np
import gym
import os
from sklearn.metrics import SCORERS
from tqdm import tqdm
import argparse

# date: 2022/6/13 19:00

total_reward = []
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="Qlearning_num_bins_4",
                    help="Determines the name of saved data, default is \"Qlearning\"")

parser.add_argument("--episode",        type=int,      default=1200,
                    help="Determines the episode we want to train per time")
parser.add_argument("--num_train",      type=int,      default=1,
                    help="Determines the times of training process")
parser.add_argument("--num_bins",       type=int,      default=4,
                    help="Number of part that the continuous space is to be sliced into, \"DO NOT\" set this to be larger than 10")

parser.add_argument("--learning_rate",  type=float,    default=0.5,
                    help="Determines the step size while moving toward a minimum of a loss function")
parser.add_argument("--decay_frequency", type=int,      default=200,
                    help="learning_rate decrease per How many episode, default= episode/6")
parser.add_argument("--decay",          type=float,    default=0.045,
                    help="reduction of learning rate per an amount of episode")

parser.add_argument("--init_epsilon",   type=float,    default=0.8)
parser.add_argument("--epsilon",        type=float,    default=0.05,
                    help="Determines the explore/expliot rate of the agent")
parser.add_argument("--GAMMA",          type=float,    default=0.97,
                    help="The discount factor (tradeoff between immediate rewards and future rewards")

parser.add_argument("--learn_threshold", type=int,      default=400)
parser.add_argument("--state_len",      type=int,      default=130)
parser.add_argument("--reward_ratio",   type=int,    default=1000)
args = parser.parse_args()
max_count = [0] * 5


class Agent():
    def __init__(self, env, epsilon=args.init_epsilon, learning_rate=args.learning_rate, GAMMA=args.GAMMA, num_bins=args.num_bins):
        """
        The agent learning how to control the action of the chicken.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        self.qtable = {}
        self.bins = self.init_bins(0, 256, self.num_bins)

    def init_bins(self, lower_bound, upper_bound, num_bins):
        """
        Slice the interval into #num_bins parts.

        lower_bound: The lower bound of the interval.
        upper_bound: The upper bound of the interval.
        num_bins: Number of parts to be sliced.
        """
        size = (upper_bound - lower_bound)/num_bins
        return np.arange(lower_bound, upper_bound, size)[1:]

    def discretize_value(self, value, bins):
        """
        given a value, return its position between interval which stored in bins

        value: The value to be discretized.
        bins: A numpy array of quantiles
        """
        return np.searchsorted(bins, value, side="right")

    def discretize_observation(self, observation):
        """
        Discretize the observation which we observed from a continuous state space.

        observation: The observation to be discretized, which is a list of 128 features:

        Returns a nparray of 128 discretized features which represents the state.
        """
        answer = []
        for i in range(len(observation)):
            answer.append(self.discretize_value(observation[i], self.bins))

        return np.array(answer)

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.
        state: A representation of the current state of the enviornment.

        action_sapce:
            '0': stay in place
            '1': go forward
            '2': go backward

        Returns the action to be evaluated.
        """
        state = np.array2string(
            state, max_line_width=args.state_len, separator='')[1:-1]
        self.qtable.setdefault(state, [0.] * self.env.action_space.n)

        value = np.random.random_sample()
        if value < self.epsilon:
            a = np.random.uniform()
            if a < 0.3:
                action = self.env.action_space.sample()
            else:
                action = 1
        else:
            action = np.argmax(self.qtable[state])

        return action

    def learn(self, state, action, reward, next_state, done, count):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        state: The state of the enviornment before taking the action.
        action: The exacuted action.
        reward: Obtained from the enviornment after taking the action.
        next_state: The state of the enviornment after taking the action.
        done: A boolean indicates whether the episode is done.
        count: how many chicken cross the road successfully

        Don't need to return anything
        """
        state = np.array2string(
            state, max_line_width=args.state_len, separator='')[1:-1]
        next_state = np.array2string(
            next_state, max_line_width=args.state_len, separator='')[1:-1]

        self.qtable[state][action] = self.qtable[state][action] * (1 - self.learning_rate) \
            + self.learning_rate * (reward * args.reward_ratio + self.gamma * max(
                self.qtable.setdefault(next_state, [0.] * self.env.action_space.n)))

        # Conditions to decide when to save your table
        global max_count
        if done:
            if np.mean(max_count) <= count:
                np.save("./Train_data/Qlearning/tables/" +
                        args.file + ".npy", self.qtable)
                print(f"save table whose score is {count}")
            inx = np.argmin(max_count)
            if max_count[inx] < count:
                max_count[inx] = count

    def check_max_Q(self):
        """
        the function calculating the max Q value of initial state(self.env.reset()).

        self: the agent itself.

        Return the max Q value of initial state(self.env.reset())
        """
        state = self.discretize_observation(self.env.reset())
        state = np.array2string(
            state, max_line_width=args.state_len, separator='')[1:-1]
        max_q = np.max(self.qtable.setdefault(
            state, [0.] * self.env.action_space.n))
        return max_q


def train(env):
    """
    Train the agent on the given environment.
    """
    training_agent = Agent(env)
    rewards = []

    step = 0
    for ep in tqdm(range(args.episode)):
        state = training_agent.discretize_observation(env.reset())
        done = False

        step += 1
        if step > args.learn_threshold:
            training_agent.epsilon = args.epsilon

        count = 0
        while True:
            action = training_agent.choose_action(state)
            next_observation, reward, done, _ = env.step(action)

            next_state = training_agent.discretize_observation(
                next_observation)
            if reward:
                count += 1

            training_agent.learn(state, action, reward,
                                 next_state, done, count)

            if done:
                if step > args.learn_threshold:
                    rewards.append(count)
                break

            state = next_state

        if (ep + 1) % args.decay_frequency == 0:
            training_agent.learning_rate -= args.decay

    total_reward.append(rewards)


if __name__ == "__main__":
    '''
    The main funtion
    '''
    env = gym.make('Freeway-v4', obs_type='ram')
    # env = gym.make('Freeway-v4', obs_type = 'ram',  render_mode='human') # with game window

    if not os.path.exists("./Train_data/Qlearning/tables/"):
        os.mkdir("./Train_data/Qlearning/tables/")

    # training section:
    for i in range(args.num_train):
        print(f"#{i + 1} training progress")
        train(env)

    if not os.path.exists("./Train_data/Qlearning/rewards/"):
        os.mkdir("./Train_data/Qlearning/rewards/")

    np.save("./Train_data/Qlearning/rewards/" +
            args.file + ".npy", np.array(total_reward))

    env.close()

import numpy as np
import gym
import os
from sklearn.metrics import SCORERS
from tqdm import tqdm
import argparse

# date: 2022/6/13 02:00

total_reward = []
parser = argparse.ArgumentParser()
parser.add_argument("--seed",           type= int,      default= 81)

parser.add_argument("--episode",        type= int,      default= 600)
parser.add_argument("--decay_frequency",type= int,      default= 100)
parser.add_argument("--num_test",       type= int,      default= 1)
parser.add_argument("--num_train",      type= int,      default= 1)

parser.add_argument("--num_bins",       type= int,      default= 2)
parser.add_argument("--state_len",      type= int,      default= 130)
parser.add_argument("--reward_ratio",   type= int,    default= 1000)

parser.add_argument("--learn_threshold",type= int,      default= 200)
parser.add_argument("--decay",          type= float,    default= 0.045)
parser.add_argument("--learning_rate",  type= float,    default= 0.5)
parser.add_argument("--init_epsilon",   type= float,    default= 0.8)
parser.add_argument("--epsilon",        type= float,    default= 0.05)
parser.add_argument("--GAMMA",          type= float,    default= 0.97)

args = parser.parse_args()
max_count = [0] * 5
best_score = 0

class Agent():
    def __init__(self, env, epsilon=args.init_epsilon, learning_rate= args.learning_rate, GAMMA=args.GAMMA, num_bins=args.num_bins):
        """
        The agent learning how to control the action of the chicken.

        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: The discount factor (tradeoff between immediate rewards and future rewards)
            num_bins: Number of part that the continuous space is to be sliced into.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        #table_shape = [self.num_bins] * 128
        #table_shape.append(self.env.action_space.n)
        #self.qtable = np.zeros(table_shape)
        self.qtable = {}
        # init_bins() is your work to implement.
        self.bins = self.init_bins(0, 256, self.num_bins)

    def init_bins(self, lower_bound, upper_bound, num_bins):
        """
        Slice the interval into #num_bins parts.

        Parameters:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            num_bins: Number of parts to be sliced.

        Returns:
            a numpy array of #num_bins - 1 .		

        Example: 
            Let's say that we want to slice [0, 10] into five parts, 
            that means we need 4 quantiles that divide [0, 10]. 
            Thus the return of init_bins(0, 10, 5) should be [2. 4. 6. 8.].

        Hints:
            1. This can be done with a numpy function.
        """
        # Begin your code
        size = (upper_bound - lower_bound)/num_bins
        return np.arange(lower_bound, upper_bound, size)[1:]
        # End your code

    def discretize_value(self, value, bins):
        """
            .

        Parameters:
            value: The value to be discretized.
            bins: A numpy array of quantiles

        returns:
            The discretized value.

        Example:
            With given bins [2. 4. 6. 8.] and "5" being the value we're going to discretize.
            The return value of discretize_value(5, [2. 4. 6. 8.]) should be 2, since 4 <= 5 < 6 where [4, 6) is the 3rd bin.

        Hints:
            1. This can be done with a numpy function.				
        """
        # Begin your code
        return np.searchsorted(bins, value, side="right")
        # End your code

    def discretize_observation(self, observation):
        """
        Discretize the observation which we observed from a continuous state space.

        Parameters:
            observation: The observation to be discretized, which is a list of 128 features:

        Returns:
            state: A list of 128 discretized features which represents the state.

        Hints:
            1. All 128 features are in continuous space.
            2. You need to implement discretize_value() and init_bins() first
            3. You might find something useful in Agent.__init__()
        """
        # Begin your code
        answer = []
        for i in range(len(observation)):
            answer.append(self.discretize_value(observation[i], self.bins))
        
        return np.array(answer)
        # End your code

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.

        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.

        Returns:
            action: The action to be evaluated.
        """
        # Begin your code
        state = np.array2string(state, max_line_width = args.state_len,separator = '')[1:-1]
        self.qtable.setdefault(state, [0.]* self.env.action_space.n)

        value = np.random.random_sample()
        if value < self.epsilon:
            a = np.random.uniform()
            if a < 0.3: action = self.env.action_space.sample()
            else: action = 1
        else:
            action = np.argmax(self.qtable[state])
            #print(self.qtable[state])
        
        return action
        # End your code

    def learn(self, state, action, reward, next_state, done, count):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.

        Returns:
            None (Don't need to return anything)
        """
        # Begin your code
        state = np.array2string(state, max_line_width = args.state_len,separator = '')[1:-1]
        next_state = np.array2string(next_state, max_line_width = args.state_len,separator = '')[1:-1]

        #print(self.qtable.get(state, "None"))
        self.qtable[state][action] = self.qtable[state][action] *(1 - self.learning_rate) \
             + self.learning_rate * (reward * args.reward_ratio + self.gamma * max(self.qtable.setdefault(next_state, [0.]* self.env.action_space.n)))
        '''
        if self.qtable[state][0] or self.qtable[state][1] or self.qtable[state][2]:
            print(self.qtable[state])
            print(self.qtable[next_state])
            print()
        '''
        # End your code

        # You can add some conditions to decide when to save your table
        global max_count
        if done:
            if np.mean(max_count) <= count: 
                np.save("./Tables/Qtable_Freeway.npy", self.qtable)
                #print("table_saved")

            inx = np.argmin(max_count)
            if max_count[inx] < count: max_count[inx] = count

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
        # Begin your code
        state = self.discretize_observation(self.env.reset())
        state = np.array2string(state, max_line_width = args.state_len,separator = '')[1:-1]
        max_q = np.max(self.qtable.setdefault(state, [0.] * self.env.action_space.n))
        return max_q
        # End your code


def train(env):
    """
    Train the agent on the given environment.

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
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

            next_state = training_agent.discretize_observation(next_observation)
            if reward: 
                count += 1
                #print("hit the road")

            training_agent.learn(state, action, reward, next_state, done, count)

            global best_score
            if done:
                rewards.append(count)
                if count > best_score:
                    best_score = count
                    print(count)
                break

            state = next_state


        if (ep + 1) % args.decay_frequency == 0:
            training_agent.learning_rate -= args.decay

    total_reward.append(rewards)


def test(env):
    """
    Test the agent on the given environment.

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)

    # Change the filename to your student id
    testing_agent.qtable = np.load("./Tables/Qtable_Freeway.npy", allow_pickle = True).item()
    rewards = []

    for _ in range(args.num_test):
        state = testing_agent.discretize_observation(testing_agent.env.reset())
        count = 0
        while True:
            state = np.array2string(state, max_line_width = args.state_len, separator = '')[1:-1]
            action = np.argmax(testing_agent.qtable.setdefault(state, [0.] * testing_agent.env.action_space.n))
            next_observation, reward, done, _ = testing_agent.env.step(action)
            if reward: count += 1
            next_state = testing_agent.discretize_observation(next_observation)

            if done == True:
                rewards.append(count)
                break

            state = next_state

    print(f"average reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    '''
    The main funtion
    '''
    # Please change to the assigned seed number in the Google sheet
    SEED = args.seed

    env = gym.make('Freeway-v4', obs_type = 'ram',  render_mode='human')
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    '''
    for i in range(args.num_train):
        print(f"#{i + 1} training progress")
        train(env)
    '''
    # testing section:
    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/Qtable_Freeway_rewards.npy", np.array(total_reward))

    env.close()

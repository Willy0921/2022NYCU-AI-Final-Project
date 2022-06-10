import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd

import matplotlib.pyplot as plt

import torch
from dqn import DQN, LearntAgent
from atari import Atari
from settings import *


def train():
    num_episodes = TRAIN_EP
    log_fd = open(LOG_FILE,'w')

    agent = DQN()
    env = Atari()

    if LOAD_MODEL:
        agent.restore_model("./model/q_target_checkpoint.pth")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    global_steps = 0
    results = {}
    results_eva = {}

    for i_episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            
            if done:
                next_state = None

            agent.memory.push(  state, \
                                action, \
                                next_state, \
                                torch.tensor([[reward]], device=DEVICE), \
                                torch.tensor([done], device=DEVICE, dtype=torch.bool))
            state = next_state
            episode_reward += reward
            global_steps += 1 

            if global_steps > 50000:
                agent.update()

            if done:
                train_info_str = f"Episode: {i_episode+1}, select_action_steps: {agent.select_action_steps}, reward: {episode_reward}, epsilon: {agent.epsilon}"
                print(train_info_str)
                agent.select_action_steps = 0
                results[i_episode] = episode_reward
                log_fd.write(train_info_str)
                break

        test_env = Atari()
        average_reward = 0
        test_ep = 5

        for _ in range(test_ep):
            state = test_env.reset()
            episode_reward = 0
            while True:
                action = agent.evaluate_action(state)
                state, reward, done, _ = test_env.step(action.item())
                episode_reward += reward
                if done:
                    break
            average_reward += episode_reward
        average_reward /= test_ep
        agent.evaluate_action_steps /= test_ep
        
        eval_info_str = f"Evaluation: True, Episode: {i_episode+1}, evaluate_action_steps: {agent.evaluate_action_steps}, evaluate reward: {average_reward}"
        results_eva[i_episode] = average_reward
        print(eval_info_str)
        agent.evaluate_action_steps = 0
        log_fd.write(eval_info_str)

    agent.save_model(SAVE_DIR)
    print(f"[Info] Save model at '{SAVE_DIR}' !")
    
    log_fd.close()
    results_df = pd.DataFrame(results.items(), columns=['Episode','Reward'])
    results_df.to_csv(r'./result/episode_reward_learning.csv', index=False)
    results_df

    results_df_1 = pd.DataFrame(results_eva.items(), columns=['Episode','Reward'])
    results_df_1.to_csv(r'./result/episode_reward_evaluation.csv', index=False)
    results_df_1

def test():
    agent = DQN()
    env = Atari()
    
    agent.restore_model("./model/q_target_checkpoint.pth")
    results = {}

    for i_episode in range(10):
        episode_reward = 0
        state = env.reset()

        while True:
            action = agent.evaluate_action(state)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode: {i_episode+1}, evaluate_action_steps: {agent.evaluate_action_steps}, reward: {episode_reward}")
                results[i_episode] = episode_reward
                agent.evaluate_action_steps = 0
                break
    results_df = pd.DataFrame(results.items(), columns=['Episode','Reward'])
    results_df.to_csv(r'./result/episode_reward_testing.csv', index=False)
    results_df

def play():
    agent = LearntAgent()
    env = Atari(render=True)
    
    agent.restore_model("./model/q_target_checkpoint.pth")
    for i_episode in range(1):
        episode_reward = 0
        state = env.reset()

        fig = plt.figure()
        fig.set_facecolor('w')

        while True:
            action, q_value = agent.evaluate_action(state)
            
            plt.title(f"NOPE: {q_value[0,0]}, UP: {q_value[0,1]}, DOWN: {q_value[0,2]}\n")
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode: {i_episode+1}, evaluate_action_steps: {agent.evaluate_action_steps}, reward: {episode_reward}")
                agent.evaluate_action_steps = 0
                break

if __name__ == "__main__":
    train()
    test()
    play()
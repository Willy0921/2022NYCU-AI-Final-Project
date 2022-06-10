import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import random
import numpy as np
from settings import *


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class CNN(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)

class DQN(object):
    def __init__(self):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPSILON_START
        self.EPS_END = EPSILON_FINAL
        self.EPS_DECAY = EPSILON_DECAY
        self.LEARN_RATE = LEARNING_RATE

        self.action_dim = 3
        self.state_dim = (84,84)
        self.epsilon = 0.0
        self.update_count = 0
        
        self.policy_net = CNN(84, 84, self.action_dim).to(device)
        self.target_net = CNN(84, 84, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LEARN_RATE)
        
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.select_action_steps = 0
        self.evaluate_action_steps = 0

    def select_action(self, state):
        self.select_action_steps += 1
        self.epsilon = self.EPS_END + np.maximum( (self.EPS_START-self.EPS_END) * (1 - self.select_action_steps/self.EPS_DECAY), 0)
        if random.random() < self.epsilon:
            return torch.tensor([random.sample([0,0,0,1,1,1,1,1,1,2],1)], device=device, dtype=torch.long)  
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def evaluate_action(self, state, rand=0.1):
        self.evaluate_action_steps += 1
        if random.random() < rand:
            return torch.tensor([random.sample([0,0,0,1,1,1,1,1,1,2],1)], device=device, dtype=torch.long)
        with torch.no_grad():
            return self.target_net(state).max(1)[1].view(1, 1)


    def store(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def update(self):
        if len(self.memory) < self.BATCH_SIZE:
            print("[Warning] Memory data less than batch sizes!")
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        final_mask = torch.cat(batch.done)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE,1, device=device)
        next_state_values[final_mask.bitwise_not()] = self.target_net(non_final_next_states).max(1, True)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
    
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % TARGET_STEP == 0:
            self.update_target_net()

        
    def update_target_net(self):
        with torch.no_grad():
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="."):
        torch.save(self.target_net.state_dict(), path+'/q_target_checkpoint.pth'.format(self.select_action_steps))
        torch.save(self.policy_net.state_dict(), path+'/q_policy_checkpoint.pth'.format(self.select_action_steps))

    def restore_model(self, path):
        self.target_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.eval()
        print("[Info] Restore model from '%s' !"%path)

class LearntAgent(object):
    def __init__(self):
        self.action_dim = 3
        self.net = CNN(84, 84, self.action_dim).to(device)
        self.evaluate_action_steps = 0

    def evaluate_action(self, state, rand=0.1):
        self.evaluate_action_steps += 1
        if random.random() < rand:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long), np.array([[0., 0., 0.,]])
        with torch.no_grad():
            q_value = self.net(state)
            return q_value.max(1)[1].view(1, 1), q_value.cpu().numpy()

    def restore_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()
        print(f"[Info] Restore model from '{path}' !")
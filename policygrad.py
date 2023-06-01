import viz_conversion
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym


# Hyperparameters
num_episodes = 1000 # My github pages is optimized for 1000 episodes
learning_rate = 0.001
gamma = 0.97
watch_play = 0
plot_rewards = 0

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # My visualization code on my github pages site is optimized for this size
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    logits = policy(state)
    m = Categorical(logits=logits)
    action = m.sample()
    log_prob = m.log_prob(action)
    log_probs_list.append(log_prob)
    return action.item()

def update_policy(rewards, log_probs):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)

    policy_loss = torch.cat(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

env = gym.make('CartPole-v1')
reward_track = []  # list to record rewards
weight_history = []

for episode in range(num_episodes):
    state, info = env.reset()
    rewards = []
    log_probs_list = []
    print(f"Episode: {episode}\n")

    for t in range(1000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, _, _ = env.step(action)
        print(f"\tt: {t}")
        rewards.append(reward)

        if done:
            update_policy(rewards, log_probs_list)
            reward_track.append(sum(rewards))  # record the total reward earned in the episode
            weights = policy.state_dict()  # Get the current model weights
            weights_numpy = {key: value.numpy() for key, value in weights.items()}  # Convert tensors to NumPy arrays
            weight_history.append(copy.deepcopy(weights_numpy))
            break

viz_conversion.convert_model_to_json(policy, weight_history)
viz_conversion.convert_rewards_to_json(reward_track)

if plot_rewards:
    plt.plot(reward_track)
    plt.title('Reward over time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def play_game():
    state, info = env.reset()
    for t in range(1000):
        env.render() # Render the environment to the screen
        action = select_action(state)
        state, reward, done, _, _ = env.step(action)
        if done:
            break

if watch_play:
    play_game()
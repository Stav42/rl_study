from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

mse_loss = nn.MSELoss()

class Action_Policy(nn.Module):

    def __init__(self, obs_space_dims: int, action_space_dims: int):

        super().__init__()

        # Shared Network for both mean and standard deviation
        self.shared_net = nn.Sequential(
            nn.Linear(in_features=29, out_features=20),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=15),
            nn.Tanh(),
        )

        # Policy Mean specific layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(in_features=15, out_features=8)
        )

        # Policy Mean specific layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(in_features=15, out_features=8)
        )

    def forward(self, x):
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    
class REINFORCE:

    def __init__(self, obs_space_dims: int, action_space_dims: int):

        self.learning_rate = 1e-4
        self.gamma = 0.95
        self.eps = 1e-6

        existing = False

        self.probs = []
        self.rewards = []
        self.values = []

        self.policy_net = Action_Policy(obs_space_dims, action_space_dims)
        if existing:
            self.policy_net.load_state_dict(torch.load('nn_policy.pth'))
        self.pol_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:

        state = torch.tensor(state)
        action_means, action_stddevs = self.policy_net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means+ self.eps, action_stddevs + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action
    
    def update(self):

        running_reward = 0
        returns = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_reward = R + self.gamma * running_reward
            returns.insert(0, running_reward)

        deltas = torch.tensor(returns)

        self.returns_val = returns

        ###### OPTIMIZE POLICY ########################
        pol_loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            pol_loss += log_prob.mean() * delta * (-1)
        
        #Update Policy Network
        self.pol_optimizer.zero_grad()
        pol_loss.backward()
        self.pol_optimizer.step()

        # for name, param in self.policy_net.named_parameters():
            # if param.grad is not None:
            #     print(name, "Policy gradient:", param.grad)

        self.probs = []
        self.rewards = []

render = 0

if render:
    env = gym.make("Ant-v4", render_mode = "human")
else:
    env = gym.make("Ant-v4")

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]

rewards_over_seeds = []

def get_reward(info):
    return -1 * np.linalg.norm(np.array([10, 0]) - np.array([info['x_position'], info['y_position']]))


for seed in [1, 2, 3, 5, 8]:
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):

        obs, info = wrapped_env.reset(seed=seed)
        done = False

        obs = np.append(obs, np.array([0, 0]))
        ok = 0
        
        while not done:
            if ok:
                obs = np.append(obs, position)
            action = agent.sample_action(obs)

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            # reward = get_reward(info)
            # print(info)
            # print("Distance reward: ", reward)
            # print("Survival reward: ", info['reward_survive'], info['reward_ctrl'])
            # reward = info['reward_survive'] + info['reward_ctrl'] + reward
            

            agent.rewards.append(reward)
            position = np.array([info['x_position'], info['y_position']])
            ok = 1
            done = terminated or truncated
        
        reward_val = agent.rewards
        reward_over_episodes.append(reward_val)
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(reward_val))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)


rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
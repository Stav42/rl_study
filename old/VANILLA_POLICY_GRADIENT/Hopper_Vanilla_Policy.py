from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import time
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

mse_loss = nn.MSELoss()


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    
class Value_Network(nn.Module):

    def __init__(self, obs_space_dims: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features=obs_space_dims, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=1)


    def forward(self, x):

        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)

        return y


class VANILLA_POLICY_GRADIENT:

    def __init__(self, obs_space_dims: int, action_space_dims: int):

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.values = []
        self.returns = []

        self.pol_net = Policy_Network(obs_space_dims, action_space_dims)
        self.val_net = Value_Network(obs_space_dims)
        existing = 1
        if existing:
            self.pol_net.load_state_dict(torch.load('hopper_VPGpol_setpt_rew20.pth'))
            self.val_net.load_state_dict(torch.load('hopper_VPGval_setpt_rew20.pth'))
        self.pol_optimizer = torch.optim.AdamW(self.pol_net.parameters(), lr=self.learning_rate)
        self.val_optimizer = torch.optim.AdamW(self.val_net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:

        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.pol_net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs) - torch.tensor(self.values)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        val_loss = 0
        for val, ret in zip(self.values, self.returns):
            val_loss+= (val - ret)**2

        # Update the policy network
        start_time = time.time()
        self.pol_optimizer.zero_grad()
        loss.backward()
        self.pol_optimizer.step()
        print("Time it took to optimize the policy: ", time.time() - start_time)

        # for name, param in self.pol_net.named_parameters():
        #     if param.grad is not None:
        #         print(name, "Policy gradient:", param.grad)

        # Update the Value Network
        self.val_optimizer.zero_grad()
        val_loss.backward()
        self.val_optimizer.step()

        # for name, param in self.val_net.named_parameters():
        #     if param.grad is not None:
        #         print(name, "Value gradient:", param.grad)

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.returns = []
        self.values = []



# Create and wrap the environment
# env = gym.make("Hopper-v4", render_mode="human", exclude_current_positions_from_observation=False, forward_reward_weight=0)
env = gym.make("Hopper-v4", exclude_current_positions_from_observation=False, forward_reward_weight = 0)

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(2.5e4)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []
timestep_over_seeds = []

forward_vel_over_seed = []
x_over_seed = []

def get_return(reward):
    running_g = 0
    gs = []

    # Discounted return (backwards) - [::-1] will return an array in reverse
    for R in reward[::-1]:
        running_g = R + 0.97 * running_g
        gs.insert(0, running_g)

    return gs

def get_reward(info, obs):
    return -20 * (info['x_position'] + 5) + 10 * obs[0]

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = VANILLA_POLICY_GRADIENT(obs_space_dims, action_space_dims)
    reward_over_episodes = []
    timesteps = 600
    timestep_count = []

    forward_vel = []
    x_pos = []


    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)
        # print(episode)
        done = False
        # env.render()
        count = 0
        for t in range(timesteps):
            action = agent.sample_action(obs)
            value = agent.val_net(torch.tensor(np.array([obs]), dtype=torch.float32))
            count+=1

            agent.values.append(value)
            # print("Timesteps: ", t)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            reward = get_reward(info, obs)
            # print(reward) 
            agent.rewards.append(reward)

            # print(obs)
            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            # done = terminated or truncated
            done = truncated

        returns = get_return(agent.rewards)
        ## Check 1 (Flipped?):
        # print("Returns: ", returns)
        agent.returns = returns

        reward_over_episodes.append(returns)
        agent.update()

        if episode % 100 == 0:
            avg_reward = float(np.mean(np.array(returns)))
            print("Episode:", episode, "Average Reward:", avg_reward)
            print("max return", np.max(np.array(returns)))
            print("min return", np.min(np.array(returns)))
            print("std dev of return: ", np.var(np.array(returns)))

        timestep_count.append(count)

    rewards_over_seeds.append(reward_over_episodes)
    timestep_over_seeds.append(timestep_count)

    torch.save(agent.pol_net.state_dict(), 'hopper_VPGpol_setpt_rew20.pth')
    torch.save(agent.val_net.state_dict(), 'hopper_VPGval_setpt_rew20.pth')



rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for Hopper-v4"
)
plt.show()
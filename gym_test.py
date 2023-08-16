import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=29, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=8)
        self.output = nn.Tanh()

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        y = self.output(y)
        return y
    
class Value(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=29, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=1)
        # self.output = nn.Sigmoid()

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        # y = self.output(y)
        return y

policy_nn = Policy()
value_nn = Value()

rate_learning = 1e-3

optim_pol = optim.Adam(policy_nn.parameters(), lr=rate_learning)
optim_val = optim.Adam(value_nn.parameters(), lr=rate_learning)


def get_reward(info):
    return -1 * np.linalg.norm(np.array([10, 0]) - np.array([info['x_position'], info['y_position']]))


# env = gym.make('Ant-v4', render_mode="human")
env = gym.make('Ant-v4')
observation, info = env.reset()
# env.render()
viewer = None
# viewer = env.env.env.env.mujoco_renderer.viewer

line_start = np.array([0, 0, 0])
line_end = np.array([10, 0, 0])

loss1 = nn.MSELoss()


# K iterations of learning
for k in range(1):
    # 100 trajectories

    states = []
    returns = []

    actions_tr = []
    rewards_tr = []
    states_tr = []

    # Calculating returns for all timesteps (1000) in 100 trajectories
    returns = np.zeros([10, 1000])
    rewards = np.zeros([10, 1000])
    states = np.zeros([10, 1000, 29])
    actions = np.zeros([10, 1000, 8])
    advantages = np.zeros([10, 1000])
    values = np.zeros([10, 1000])

    for traj in range(10):
        # 1000 timesteps
        print("Traj is: ", traj)
        for t in range(1000):

            if t == 0:
                observation = np.append(observation, np.array([0, 0]))
            else:
                observation = np.append(observation, np.array([info['x_position'], info['y_position']]))

            states[traj, t] = observation

            action = policy_nn(torch.from_numpy(observation).float())
            action = action.detach().numpy()
            value_i = value_nn(torch.from_numpy(observation).float())
            value_i = value_i.detach().numpy()
            
            observation, reward, terminated, truncated, info = env.step(action)
            reward = get_reward(info)

            advantage = reward - value_i

            rewards[traj, t] =  reward
            actions[traj, t] = action
            advantages[traj, t] = advantage
            values[traj, t] = value_i
            
            if viewer is not None:
                viewer.add_marker(pos=line_start, size=0.2, rgba=(255, 0, 0, 1))
                viewer.add_marker(pos=line_end, size=0.2, rgba=(255, 0, 0, 1))
                # print("Marked")

            if terminated or truncated:
                observation, info = env.reset()


        for i, reward in enumerate(rewards):
            return_i = np.cumsum(reward)
            returns[i] = np.flip(return_i)

    # Ran through all the trajectories and collected data

    # Keep values as a tensor with gradient tracking enabled
    values_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Flatten the tensors
    values_flatten = values_tensor.view(-1)
    rewards_flatten = rewards_tensor.view(-1)

    loss = loss1(values_flatten, rewards_flatten)

    optim_pol.zero_grad()
    loss.backward()
    optim_pol.step()


    ## Now improving the policy:
    

env.close() 
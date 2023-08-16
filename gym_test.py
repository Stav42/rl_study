import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=29, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=8)
        self.output = nn.Tanh()
        # Log standard deviations for each action dimension
        self.log_std = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        mean = self.output(y)
        std = torch.exp(self.log_std)
        return mean, std
        
class Value(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=29, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=1)


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


env = gym.make('Ant-v4', render_mode="human")
# env = gym.make('Ant-v4')
observation, info = env.reset()
env.render()
viewer = None
# viewer = env.env.env.env.mujoco_renderer.viewer

line_start = np.array([0, 0, 0])
line_end = np.array([10, 0, 0])

loss1 = nn.MSELoss()
N_traj = 10

# K iterations of learning
for k in range(1):
    # 100 trajectories

    states = []
    returns = []

    actions_tr = []
    rewards_tr = []
    states_tr = []

    # Calculating returns for all timesteps (1000) in 100 trajectories
    returns = np.zeros([N_traj, 1000])
    rewards = np.zeros([N_traj, 1000])
    states = np.zeros([N_traj, 1000, 29])
    actions = np.zeros([N_traj, 1000, 8])
    advantages = np.zeros([N_traj, 1000])
    values = np.zeros([N_traj, 1000])
    actions_log = np.zeros([N_traj, 1000])

    accumulated_gradients = [torch.zeros_like(param) for param in policy_nn.parameters()]


    for traj in range(N_traj):
        # 1000 timesteps
        observation, info = env.reset()

        print("Traj is: ", traj)
        for t in range(1000):

            print(t)

            if t == 0:
                observation = np.append(observation, np.array([0, 0]))
            else:
                print(info)
                print(info['x_position'])

                observation = np.append(observation, np.array([info['x_position'], info['y_position']]))

            states[traj, t] = observation

            action_mean, action_std = policy_nn(torch.from_numpy(observation).float())
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            action_log = action_dist.log_prob(action)
            # action = action.detach().numpy()
            value_i = value_nn(torch.from_numpy(observation).float())
            value_i = value_i.detach().numpy()
            
            action = action.detach().numpy()

            observation, reward, terminated, truncated, info = env.step(action)
            reward = get_reward(info)

            advantage = reward - value_i

            rewards[traj, t] =  reward
            advantages[traj, t] = advantage
            values[traj, t] = value_i
            actions_log[traj, t] = action_log.sum()
            
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
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, requires_grad=True)
    actions_log_tensor = torch.tensor(actions_log, dtype=torch.float32, requires_grad=True)

    # Flatten the tensors
    values_flatten = values_tensor.view(-1)
    rewards_flatten = rewards_tensor.view(-1)

    loss = loss1(values_flatten, rewards_flatten)

    optim_val.zero_grad()
    loss.backward()
    optim_val.step()

    ## Now improving the policy:

    for traj in range(N_traj):
        for t in range(1000):

            loss_t = -(actions_log[traj, t])*returns[traj, t]

            loss_t = torch.tensor(loss_t, dtype=torch.float32, requires_grad=True)
            loss_t.backward(retain_graph=True)

            for param, acc_grad in zip(policy_nn.parameters(), accumulated_gradients):
                if param.grad is not None:
                    acc_grad += param.grad
            
            optim_pol.zero_grad()


    for acc_grad in accumulated_gradients:
        acc_grad /= (N_traj * 1000)

    for param, acc_grad in zip(policy_nn.parameters(), accumulated_gradients):
        param.data -= rate_learning * acc_grad

    # gradients = None
    
    # for traj in range(N_traj):
    #     grad_val = -1 * np.multiply(actions_log[traj], advantages[traj])
    #     grad_val_tensor = torch.tensor(grad_val, dtype=torch.float32, requires_grad=True)
    #     grad_val_tensor.backkward()
    #     if gradients is not None:
    #         gradients += [param.grad for param in policy_nn.parameters()]
    #     else:
    #         gradients = [param.grad for param in policy_nn.parameters()]
        
    # gradients = gradients/N_traj



    

    # grad_val.backward()

    # grad_val = np.multiply(actions_log, advantages)
    # print(len(grad_val))

env.close() 
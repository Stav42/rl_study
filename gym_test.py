import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

timesteps = 200
save_model = 1

class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=29, out_features=15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=15, out_features=8)
        # self.output = nn.Tanh()
        # Log standard deviations for each action dimension
        self.log_std = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        # mean = self.output(y)
        mean = y
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

rate_learning = 5e-3

existing = 1
render = 1

if existing:
    policy_nn.load_state_dict(torch.load('policy_nn.pth'))
    value_nn.load_state_dict(torch.load('value_nn.pth'))
    print("loaded")

optim_pol = optim.Adam(policy_nn.parameters(), lr=rate_learning)
optim_val = optim.Adam(value_nn.parameters(), lr=rate_learning)


def get_reward(info):
    return -1 * np.linalg.norm(np.array([10, 0]) - np.array([info['x_position'], info['y_position']]))

env = gym.make('Ant-v4')

if render:
    env = gym.make('Ant-v4', render_mode="human")

observation, info = env.reset()

if render:
    env.render()
# env.render()
viewer = None
if render:
    viewer = env.env.env.env.mujoco_renderer.viewer

line_start = np.array([0, 0, 0])
line_end = np.array([10, 0, 0])

loss1 = nn.MSELoss()
N_traj = 10

# K iterations of learning
for k in range(500):
    # 100 trajectories

    print("iteration no: ", k)

    states = []
    returns = []

    actions_tr = []
    rewards_tr = []
    states_tr = []

    # Calculating returns for all timesteps (timesteps) in 100 trajectories
    returns = np.zeros([N_traj, timesteps])
    rewards = np.zeros([N_traj, timesteps])
    states = np.zeros([N_traj, timesteps, 29])
    actions = np.zeros([N_traj, timesteps, 8])
    advantages = np.zeros([N_traj, timesteps])
    values = np.zeros([N_traj, timesteps])
    actions_log = torch.zeros([N_traj, timesteps], requires_grad=False)

    accumulated_gradients = [torch.zeros_like(param) for param in policy_nn.parameters()]

    action_log_loss = 0
    action_log_list = []

    action_list = []
    value_list = []

    for traj in range(N_traj):
        # timesteps timesteps
        observation, info = env.reset()

        # print("Traj is: ", traj)
        for t in range(timesteps):

            if t == 0:
                observation = np.append(observation, np.array([0, 0]))
            else:
                observation = np.append(observation, position)

            states[traj, t] = observation
            action_mean, action_std = policy_nn(torch.from_numpy(observation).float())
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            action_log = action_dist.log_prob(action)
            
            action_log_list.append(action_log.sum())
            action = torch.tanh(action)
            # action = action.detach().numpy()
            value_i = value_nn(torch.from_numpy(observation).float())
            value_list.append(value_i)
            
            action = action.detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            position = np.array([info['x_position'], info['y_position']])

            reward = get_reward(info)

            advantage = reward - value_i

            rewards[traj, t] =  reward
            advantages[traj, t] = advantage
            values[traj, t] = value_i
            actions_log[traj, t] = action_log.sum().item()
            
            if viewer is not None:
                viewer.add_marker(pos=line_start, size=0.2, rgba=(255, 0, 0, 1))
                viewer.add_marker(pos=line_end, size=0.2, rgba=(255, 0, 0, 1))
                # print("Marked")

            if terminated or truncated:
                observation, info = env.reset()


        for i, reward in enumerate(rewards):
            reward = np.flip(reward)
            return_i = np.cumsum(reward)
            returns[i] = np.flip(return_i)


    # Keep values as a tensor with gradient tracking enabled
    values_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, requires_grad=True)
    actions_log_tensor = torch.tensor(actions_log, dtype=torch.float32, requires_grad=True)

    # Flatten the tensors
    advantages_flatten = (torch.tensor(returns) - torch.tensor(values)).view(-1)

    mean = torch.mean(advantages_flatten)
    std = torch.std(advantages_flatten)

    advantages_normal = (advantages_flatten - mean) / std

        # Flatten the tensors
    values_flatten = values_tensor.view(-1)
    rewards_flatten = rewards_tensor.view(-1)

    values_ten = torch.stack(value_list)
    loss = loss1(values_ten, rewards_flatten)

    optim_val.zero_grad()
    loss.backward()
    optim_val.step()

    # for name, param in value_nn.named_parameters():
    #     if param.grad is not None:
            # print(name, "Value gradient:", param.grad)

    action_log_tensor_papi = torch.stack(action_log_list)
    weighted_action_log_tensor = action_log_tensor_papi * advantages_normal
    total_loss = weighted_action_log_tensor.sum()

    # Compute gradients
    optim_pol.zero_grad()

    total_loss.backward()

    # for name, param in policy_nn.named_parameters():
    #     print(name, param.data)

    # for name, param in policy_nn.named_parameters():
    #     if param.grad is not None:
    #         print(name, "Policy gradient:", param.grad)

    # Update parameters
    optim_pol.step()

    # for name, param in policy_nn.named_parameters():
    #     print(name, param.data)

if save_model:
    torch.save(policy_nn.state_dict(), 'policy_nn.pth')
    torch.save(value_nn.state_dict(), 'value_nn.pth')
  

env.close() 
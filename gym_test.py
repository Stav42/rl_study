import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

timesteps = 100
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

rate_learning = 1e-3

existing = 0
render = 0

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
for k in range(100):
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
            action = torch.tanh(action)
            # action = action.detach().numpy()
            value_i = value_nn(torch.from_numpy(observation).float())
            value_i = value_i.detach().numpy()
            
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

    # Initialize total policy loss
    total_policy_loss = torch.tensor(0.0, dtype=torch.float32)

    for traj in range(N_traj):
        for t in range(timesteps):
            # loss_t = -(actions_log[traj, t])*(returns[traj, t] - values[traj, t])
            loss_t = -actions_log_tensor[traj, t] * (returns[traj, t] - values[traj, t])
            # Convert loss_t to a tensor with requires_grad=True
            total_policy_loss += loss_t
            

    optim_pol.zero_grad()
    total_policy_loss.backward()

    # Print gradients for debugging
    for name, param in policy_nn.named_parameters():
        if param.grad is not None:
            print(name, "gradient:", param.grad)

    optim_pol.step()

    print("Policy Loss:", total_policy_loss.item())
    for name, param in policy_nn.named_parameters():
        print(name, param.data)

    # for acc_grad in accumulated_gradients:
    #     acc_grad /= (N_traj)

    # for param, acc_grad in zip(policy_nn.parameters(), accumulated_gradients):
    #     # print("Gradient is: ", acc_grad)
    #     param.data -= rate_learning * acc_grad

    # print("Policy Parameters are: \n")

    # for name, param in policy_nn.named_parameters():
    #     print(name, param.data)

    # print("Value Parameters are: \n")

    # for name, param in value_nn.named_parameters():
    #     print(name, param.data)


if save_model:
    torch.save(policy_nn.state_dict(), 'policy_nn.pth')
    torch.save(value_nn.state_dict(), 'value_nn.pth')
  

env.close() 
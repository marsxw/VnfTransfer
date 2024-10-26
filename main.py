# %%
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import math
import copy
from collections import Counter
import networkx as nx
from environment import PhysicalNetwork
import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.mask = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, mask):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.mask[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.mask[ind]).to(device)
        )

    def __len__(self):
        return len(self.size)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        return self.linear3(x1)


class DeterministicPolicy(nn.Module,):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return F.tanh(self.mean(x))


class DDPG(object):
    def __init__(self, obs_dim, act_dim, hidden_size, gamma, tau,   expl_noise=0.1):
        self.tau = tau
        self.gamma = gamma

        # Critic网络
        self.critic = QNetwork(obs_dim, act_dim, hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        # 目标Critic网络
        self.critic_target = QNetwork(obs_dim, act_dim, hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 策略网络
        self.policy = DeterministicPolicy(obs_dim, act_dim, hidden_size)
        self.policy_optim = Adam(self.policy.parameters(), lr=1e-4)

        # 目标策略网络
        self.policy_target = DeterministicPolicy(obs_dim, act_dim, hidden_size)
        self.policy_target.load_state_dict(self.policy.state_dict())

        # 探索噪声
        self.expl_noise = expl_noise

    def select_action(self, state, evaluate=False):
        '''
            evaluate: 训练为False, 进行探索，验证时设置为True，不添加探索噪声
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.policy(state)

        if not evaluate and torch.rand(1).item() < self.expl_noise:
            action = 2 * torch.rand_like(action) - 1  # 随机采样

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):
        if replay_buffer.size < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)

        #### Critic更新 ####
        with torch.no_grad():
            next_action = self.policy_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * self.critic_target(next_state_batch, next_action)

        q_value = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q_value, next_q_value)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        #### 策略更新 ####
        policy_loss = -self.critic(state_batch, self.policy(state_batch)).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)


# %%
env = PhysicalNetwork('Chinanet.gml', sfc_num=6, overload_threshold=0.85, seed=seed)

obs_dim = env.reset().shape[0]
act_dim = 3
agent = DDPG(obs_dim,
             act_dim,
             hidden_size=256,
             gamma=.99,
             tau=.005,
             expl_noise=.8
             )

replay_buffer = ReplayBuffer(obs_dim, act_dim)


def convert_action2index(a, max_val):
    a = (a + 1) / 2  # 将 -1 ~ 1 映射到 0 ~ 1
    a = math.floor(a*max_val)
    return min(a, max_val - 1)

# %%


writer = SummaryWriter(f'runs/DDPG1_{seed}')
for episode in range(1000000):
    state = env.reset()
    episode_reward = 0

    for step in range(1000):
        action = agent.select_action(state)
        s_, d_, u_ = action
        s = convert_action2index(s_, 6)
        d = convert_action2index(d_, 6)
        u = convert_action2index(u_, len(env.G.nodes))
        next_state, reward, done, info = env.step([s, d, u])
        replay_buffer.add(state, action, reward, next_state, mask=float(not done))
        state = next_state
        episode_reward += reward

        if done or info['is_overload'] == False:
            break

    # Update the parameters of the agent
    for i in range(step):
        agent.update_parameters(replay_buffer, 256)
    agent.expl_noise *= 0.999
    if agent.expl_noise < 0.05:
        agent.expl_noise = 0.05

    writer.add_scalar('reward', episode_reward, episode)
    writer.add_scalar('step len',  step, episode)
    print(f"Episode: {episode}, Total Reward: {episode_reward}  ", 'step len',  step)

# %%
# env.step([5,5,0])
while 1:
    env.reset()
    for sfc in env.sfcs:
        for vnf in sfc.chain:
            if vnf.deployed_node == None:
                print(vnf.deployed_node)

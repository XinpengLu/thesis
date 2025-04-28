import gym
import d4rl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 加载HalfCheetah环境
env = gym.make('halfcheetah-medium-v2')

# 获取数据集
dataset = env.get_dataset()

# 提取状态、动作、奖励、下一个状态、是否终止
states = dataset['observations']
actions = dataset['actions']
rewards = dataset['rewards']
next_states = dataset['next_observations']
terminals = dataset['terminals']  # 终止标志 (0: 未终止, 1: 终止)

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 连接 (s, a)
        return self.model(x)


# 转换数据为 Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)

# 初始化模型
reward_model = RewardModel(state_dim=states.shape[1], action_dim=actions.shape[1]).to(device)
optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练循环
for epoch in range(100):
    pred_rewards = reward_model(states_tensor, actions_tensor)
    loss = loss_fn(pred_rewards, rewards_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

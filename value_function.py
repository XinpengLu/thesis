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


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.model(state)


# 转换数据为 Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)

# 初始化 V(s) 网络
value_net = ValueNetwork(state_dim=states.shape[1]).to(device)
optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 折扣因子
gamma = 0.99

# 训练循环
for epoch in range(100):
    V_s = value_net(states_tensor)
    with torch.no_grad():
        V_next_s = value_net(torch.tensor(next_states, dtype=torch.float32, device=device))
        target_values = rewards_tensor + gamma * V_next_s * (
                    1 - torch.tensor(terminals, dtype=torch.float32, device=device).unsqueeze(-1))

    loss = loss_fn(V_s, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

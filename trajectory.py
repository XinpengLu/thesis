import d4rl
import gym
import numpy as np
import torch
from torch import nn

from network import FullyConnectedQFunction

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

def load_critic(critic_path):
    critic = FullyConnectedQFunction(observation_space[0], action_space[0], True, 5)
    checkpoint = torch.load(critic_path)
    critic.load_state_dict(state_dict=checkpoint["critic1"])
    critic.eval()
    return critic.to("cuda")


# 加载 AntMaze-umaze-v2 环境
env = gym.make("antmaze-umaze-v2")
dataset = d4rl.qlearning_dataset(env)  # 获取离线数据集
observation_space = env.observation_space.shape
action_space = env.action_space.shape
max_action = float(env.action_space.high[0])

trajectories = []
current_traj = []
for i in range(len(dataset["observations"])):
    s, a, r, s_next, done = (dataset["observations"][i], dataset["actions"][i],
                             dataset["rewards"][i], dataset["next_observations"][i],
                             dataset["terminals"][i])
    current_traj.append((s, a, r, s_next))
    if done:
        trajectories.append(current_traj)
        current_traj = []

print(len(trajectories))
state_to_traj_info = {}
for traj in trajectories:
    T = len(traj)
    for t, (s, a, r, s_next) in enumerate(traj):
        state_to_traj_info[tuple(s)] = (T, t)
print("666")
print(len(state_to_traj_info))
state = dataset["observations"][4000]
# print(state_to_traj_info)

gamma = 0.99
T, t = state_to_traj_info[tuple(state)]
print(T, t)
gamma_weight = gamma ** (T - t)  # 计算折扣因子
print(gamma_weight)

# print("轨迹数量:", len(trajectories))
# goal_q_values = []
# i = 0
# for traj in trajectories:
#     if i % 1000 == 0:
#         print(i)
#     s_goal = torch.tensor(traj[-1][0], dtype=torch.float32).to('cuda')  # 终点状态
#     a_goal = torch.tensor(traj[-1][1], dtype=torch.float32).to('cuda')  # 终点动作
#     q_function = load_critic('offline_policy_checkpoints/Cal-QL-antmaze-umaze-v2/checkpoint.pt')
#     q_goal = q_function(s_goal, a_goal).detach().cpu().numpy()  # 计算最优策略 Q 值
#     goal_q_values.append(q_goal)
#     i = i + 1
#
# # 3. 计算轨迹成功概率 (Softmax 归一化)
# goal_q_values = np.array(goal_q_values)
# success_prob = np.exp(goal_q_values) / np.sum(np.exp(goal_q_values))  # 归一化成功率
# print(success_prob)
# print(success_prob.shape)
# # 4. 按成功率重新采样轨迹
# num_samples = 10  # 采样轨迹数量
# selected_trajectories = np.random.choice(trajectories, size=num_samples, p=success_prob)


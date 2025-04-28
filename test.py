import argparse
import os

import gym
import d4rl  # 需要安装 d4rl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from stable_baselines3 import SAC

from configs.command_parser import command_parser, merge_args
from utils import compute_mean_std, normalize_states, wrap_env, set_seed


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    # parse arguments
    parser = command_parser()
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    config = merge_args(args, config)
    return args, config


def plot(x_positions, y_positions, N):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_positions, y_positions, s=1, label="Agent Trajectory", c=np.linspace(0, 1, N), cmap="cool")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("AntMaze U-Maze Trajectory")
    plt.colorbar(label="Time Step Progression")
    plt.legend()
    plt.grid(True)
    plt.show()


# 选择环境
args, config = parse_args_and_config()
setattr(config, 'pid', str(os.getpid()))
env = gym.make(config.dataset.name)

# 'antmaze-umaze-v2'
dataset = d4rl.qlearning_dataset(env)
seed = 42
state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)
set_seed(seed, env)

# 目标数据集大小
syn_dset_size = 2048

# observations, actions, rewards, terminals = dataset["observations"], dataset["actions"], dataset["rewards"], dataset[
#     "terminals"]
#
# # 1. 划分完整轨迹
# trajectories, current_traj = [], {"observations": [], "actions": [], "rewards": [], "terminals": []}
#
# for i in range(len(observations)):
#     for key in current_traj:
#         current_traj[key].append(locals()[key][i])  # 动态填充数据
#
#     if terminals[i]:  # 轨迹结束
#         trajectories.append(current_traj)
#         current_traj = {k: [] for k in current_traj}  # 重新初始化
#
# # 2. 选择成功轨迹并按长度升序排序
# successful_trajectories = sorted(
#     [traj for traj in trajectories if np.sum(traj["rewards"]) > 0 and len(traj["observations"]) > 1],
#     key=lambda traj: len(traj["observations"])
# )
#
# # 3. 累积数据，直到达到 `syn_dset_size`
# selected_observations, selected_actions = [], []
# total_count = 0
#
# for traj in successful_trajectories:
#     remain = min(syn_dset_size - total_count, len(traj["observations"]))  # 计算还需要多少数据点
#     selected_observations.extend(traj["observations"][:remain])
#     selected_actions.extend(traj["actions"][:remain])
#     total_count += remain
#     if total_count >= syn_dset_size:
#         break  # 达到目标数据量


observations, actions, rewards, terminals = dataset["observations"], dataset["actions"], dataset["rewards"], dataset[
    "terminals"]

# 1. 划分完整轨迹
trajectories, current_traj = [], {"observations": [], "actions": [], "rewards": [], "terminals": []}

for i in range(len(observations)):
    for key in current_traj:
        current_traj[key].append(locals()[key][i])  # 填充数据

    if terminals[i]:  # 轨迹结束
        trajectories.append(current_traj)
        current_traj = {k: [] for k in current_traj}  # 重新初始化

# 2. 选择成功轨迹
successful_trajectories = [traj for traj in trajectories if
                           np.sum(traj["rewards"]) > 0 and len(traj["observations"]) > 1]

# 3. 提取轨迹的起始状态（或均值状态）用于 K-Means 聚类
trajectory_representations = np.array([traj["observations"][0] for traj in successful_trajectories])

# 4. 进行 K-Means++ 聚类，确保多样性
num_clusters = min(syn_dset_size, len(trajectory_representations))
kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0).fit(trajectory_representations)
cluster_labels = kmeans.labels_

# 5. 在每个聚类中选择最佳轨迹
selected_trajectories = []
for i in range(num_clusters):
    cluster_trajs = [traj for traj, label in zip(successful_trajectories, cluster_labels) if label == i]
    best_traj = max(cluster_trajs, key=lambda traj: len(traj["observations"]))  # 可改成 np.sum(traj["rewards"])
    selected_trajectories.append(best_traj)

# 6. 累积数据，直到 `syn_dset_size`
selected_observations, selected_actions = [], []
total_count = 0

for traj in selected_trajectories:
    remain = min(syn_dset_size - total_count, len(traj["observations"]))
    selected_observations.extend(traj["observations"][:remain])
    selected_actions.extend(traj["actions"][:remain])
    total_count += remain
    if total_count >= syn_dset_size:
        break  # 达到目标数据量

# 转换为 PyTorch Tensor 并存入 self.observations.weight 和 self.actions.weight
observations = np.array(selected_observations)
actions = np.array(selected_actions)

print(f"最终选取 {total_count} 个数据点 (目标: {syn_dset_size})")
print(observations.shape)
print(actions.shape)
plot(observations[:, 0], observations[:, 1], syn_dset_size)


def load_critic(env, critic_path):
    dummy_model = SAC("MlpPolicy", env)
    checkpoint = torch.load(critic_path)
    dummy_model.critic.load_state_dict(checkpoint)
    dummy_model.critic.eval()
    return dummy_model.critic.to("cuda")


def load_policy(env, policy_path):
    model = SAC("MlpPolicy", env)
    checkpoint = torch.load(policy_path)
    model.policy.load_state_dict(checkpoint)
    model.policy.eval()
    return model.policy.to("cuda")


# ****************************************************
offline_policy_path = 'q_pi_star.pt'
offline_critic = load_critic(env, offline_policy_path)
q1, q2 = offline_critic(
    torch.Tensor(observations).to('cuda'),
    torch.Tensor(actions).to('cuda')
)
q_value = torch.min(q1, q2)
print(q_value)
print(q_value.shape)

# *******************************************************
policy_path = "pi_star.pt"
policy = load_policy(env, policy_path)
with torch.no_grad():
    action = policy(torch.Tensor(observations).to('cuda'))
print(action)
print(action.shape)

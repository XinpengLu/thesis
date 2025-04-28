import argparse
import os

import gym
import d4rl  # 需要安装 d4rl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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
# 获取状态空间和动作空间
print("State shape:", env.observation_space.shape)
print("Action shape:", env.action_space.shape)

# 获取数据集
dataset = d4rl.qlearning_dataset(env)
print("Dataset keys:", dataset.keys())  # 包括 observations, actions, rewards, terminals 等

# 动作
print("State space shape:", env.observation_space.shape)
print("Action space shape:", env.action_space.shape)

# 获取轨迹数据
seed = 42
state_mean, state_std = 0, 1
dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)
set_seed(seed, env)
eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)
save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])
print(eval_intervals)
it = eval_intervals[-2]


observations = dataset["observations"]  # 形状: (N, 29) (AntMaze 的状态)

# 取出 x, y 位置(AntMaze环境的前两个状态变量是位置信息), 2-3是角速度, 4-28是关节信息
x_positions = observations[:200, 0]
y_positions = observations[:200, 1]
plot(x_positions, y_positions, 200)

# save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(
#         config.synset_size) + '_' + config.match_objective + '-q-weight_init_' + str(
#         config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')

save_path_name = 'antmaze-umaze_size_2048_offline_policy_q-weight_0.02_init_real_iter_49000_seed_0.pt'
synset = torch.load(save_path_name)

print("Observations:\n", synset.observations.weight.data)  # 256 x 17
print(synset.observations.weight.data.shape)
print("Actions:\n", synset.actions.weight.data)  # 256 x 6
print(synset.actions.weight.data.shape)

obs_array = synset.observations.weight.data.cpu().numpy()
act_array = synset.actions.weight.data.cpu().numpy()

x_positions = obs_array[:, 0]
y_positions = obs_array[:, 1]

# 可视化轨迹
plot(x_positions, y_positions, len(obs_array))

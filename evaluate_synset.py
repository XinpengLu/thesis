# 评估合成数据集
import os
import numpy as np
import time
import yaml
import argparse
import gym
import wandb
import uuid
import torch
from torch import nn
import torch.utils.data as tdata

from dataclasses import asdict
from data_load import D4RLDataset
import d4rl.gym_mujoco

from network import TanhGaussianPolicy
from tools.evaluator import Evaluator
from configs.command_parser import command_parser, merge_args

from utils import get_optimizer, set_seed, get_hms, wrap_env, compute_mean_std, normalize_states
import warnings

warnings.filterwarnings('ignore')


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


# 解析配置文件，获取训练参数
args, config = parse_args_and_config()
setattr(config, 'pid', str(os.getpid()))


# Init Wandb
def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# 加载强化学习环境（Gym+D4RL数据集）
env = gym.make(config.dataset.name)
dataset = env.get_dataset()

# 归一化环境数据
if config.normalize:
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
else:
    state_mean, state_std = 0, 1

dataset["observations"] = normalize_states(
    dataset["observations"], state_mean, state_std
)
dataset["next_observations"] = normalize_states(
    dataset["next_observations"], state_mean, state_std
)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)

# 加载评估器，用于计算策略的累计奖励
evaluator = Evaluator(env=env, config=config)
eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)

save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])

wandb_init(config)

time_start = time.time()

over_time_avg_return_list = []
over_time_avg_normalize_return_list = []

# 遍历多个迭代点的合成数据集 synset
# 从不同训练步数（iteration）的 synset 进行策略训练
for it in eval_intervals[:-2]:
    print("%s iterations" % (it))
    h, m, s = get_hms(time.time() - time_start)
    print("Execute time: %dh %dm %ds" % (h, m, s))

    # Load synset
    if config.q_weight:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(
            config.synset_size) + '_' + config.match_objective + '_q-weight_' + str(
            config.beta_weight) + '_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(
            config.seed) + '.pt')
    else:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(
            config.synset_size) + '_' + config.match_objective + '_no-q-weight_init_' + str(
            config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')

    if os.path.exists(save_path_name):
        synset = torch.load(save_path_name)

        # 评估回报值
        return_list = []
        normalize_return_list = []

        # 评估策略在环境中的表现（10 次运行取平均）
        for i in range(10):
            trajectory_return_info = evaluator.trajectory_return(synset)
            episode_return = trajectory_return_info["episode_rewards"]
            return_list.append(episode_return)
            normalize_return_list.append(env.get_normalized_score(episode_return))

        print(np.mean(return_list))
        wandb.log({"avg_episode_rewards": np.mean(return_list)}, step=it)
        wandb.log({"avg_normalize_episode_rewards": np.mean(normalize_return_list)}, step=it)

        over_time_avg_return_list.append(np.mean(return_list))
        over_time_avg_normalize_return_list.append(np.mean(normalize_return_list))

        # 计算平滑奖励
        if config.q_weight:
            # 如果Q值加权，取最近5次评估的均值
            if len(over_time_avg_return_list) > 4:
                wandb.log({"sooth_avg_episode_rewards": np.mean(over_time_avg_return_list[-5:])}, step=it)
                wandb.log({"sooth_avg_normalize_episode_rewards": np.mean(over_time_avg_normalize_return_list[-5:])},
                          step=it)
        else:
            # 取最近10次评估的均值
            if len(over_time_avg_return_list) > 10:
                wandb.log({"sooth_avg_episode_rewards": np.mean(over_time_avg_return_list[-10:])}, step=it)
                wandb.log({"sooth_avg_normalize_episode_rewards": np.mean(over_time_avg_normalize_return_list[-10:])},
                          step=it)
    else:
        continue

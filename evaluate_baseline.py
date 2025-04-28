# 评估合成数据集
import os
import sys

import numpy as np
import pandas as pd
import yaml
import argparse
import gym
import torch
from torch import nn
import torch.utils.data as tdata
from dataclasses import asdict
from data_load import D4RLDataset
import d4rl.gym_mujoco
from network import TanhGaussianPolicy
from tools.evaluator import Evaluator
from configs.command_parser import command_parser, merge_args
from utils import wrap_env, compute_mean_std, normalize_states, set_seed
import warnings

warnings.filterwarnings('ignore')


class Net:
    def __init__(
            self,
            syn_dset_size,
            observation_space,
            action_space,
    ):
        super(Net, self).__init__()
        self.syn_dset_size = syn_dset_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.observations = nn.Embedding(self.syn_dset_size, np.prod(self.observation_space))
        self.actions = nn.Embedding(self.syn_dset_size, np.prod(self.action_space))


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
    parser = command_parser()
    args = parser.parse_args()
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)
    config = merge_args(args, config)
    return args, config


def evaluate_and_print(merged, synset, evaluator, env):
    return_list = []
    normalize_return_list = []
    for i in range(10):
        trajectory_return_info = evaluator.trajectory_return_merge(
            synset) if merged else evaluator.trajectory_return(synset)
        episode_return = trajectory_return_info["episode_rewards"]
        return_list.append(episode_return)
        normalized_return = env.get_normalized_score(episode_return)
        normalize_return_list.append(normalized_return)
        print(i, normalized_return)
    return return_list, normalize_return_list


def main():
    args, config = parse_args_and_config()
    setattr(config, 'pid', str(os.getpid()))
    merged = False

    seeds = [0, 1, 2, 3, 4]
    baselines = ['herding', 'kcenter', 'kmeans_plus', 'qvalue', 'random', 'reward']
    results = {baseline: {} for baseline in baselines}

    for seed in seeds:
        for baseline in baselines:
            test_path = f'baseline/{config.dataset.name}/{seed}/{baseline}_{config.dataset.name}_256.pt'
            print(f"Evaluation env:{config.dataset.name}, baseline: {baseline}, seed:{seed}")
            env = gym.make(config.dataset.name)
            dataset = env.get_dataset()

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
            set_seed(seed, env)
            evaluator = Evaluator(env=env, config=config)

            return_list = []
            normalize_return_list = []

            if os.path.exists(test_path):
                synset = torch.load(test_path)
                return_list, normalize_return_list = evaluate_and_print(merged, synset, evaluator, env)

            out = np.mean(normalize_return_list) * 100
            results[baseline][seed] = out

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = seeds
    df['mean'] = df.mean(axis=1)
    df['std'] = df[seeds].std(axis=1)

    folder_path = f'baseline/{config.dataset.name}'
    os.makedirs(folder_path, exist_ok=True)
    excel_path = os.path.join(folder_path, f'{config.dataset.name}.xlsx')
    df.to_excel(excel_path)

if __name__ == '__main__':
    sys.exit(main())

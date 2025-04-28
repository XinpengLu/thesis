# 评估合成数据集
import os
import sys

import numpy as np
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


def merge(test_path, task1_path, task2_path=None):
    syn1, syn2 = torch.load(test_path), torch.load(task1_path)
    size1, size2 = syn1.observations.weight.shape[0], syn2.observations.weight.shape[0]
    obd_dim1, obd_dim2 = syn1.observations.weight.shape[1], syn2.observations.weight.shape[1]
    act_dim1, act_dim2 = syn1.actions.weight.shape[1], syn2.actions.weight.shape[1]

    if task2_path:
        syn3 = torch.load(task2_path)
        size3 = syn3.observations.weight.shape[0]
        obd_dim3 = syn3.observations.weight.shape[1]
        act_dim3 = syn3.actions.weight.shape[1]
    else:
        size3 = 0
        obd_dim3 = 0
        act_dim3 = 0

    merge_obd_dim = obd_dim1 + obd_dim2 + obd_dim3
    merge_act_dim = act_dim1 + act_dim2 + act_dim3
    merge_size = size1 + size2 + size3
    merge_syn = Net(merge_size, merge_obd_dim, merge_act_dim)

    def zero_pad(tensor, pad_left, pad_right):
        padding_left = torch.zeros(tensor.shape[0], pad_left, device='cuda')
        padding_right = torch.zeros(tensor.shape[0], pad_right, device='cuda')
        return torch.cat((padding_left, tensor, padding_right), dim=1)

    syn1_obs_padded = zero_pad(syn1.observations.weight, pad_left=merge_obd_dim - obd_dim1, pad_right=0)
    syn1_act_padded = zero_pad(syn1.actions.weight, pad_left=merge_act_dim - act_dim1, pad_right=0)
    syn2_obs_padded = zero_pad(syn2.observations.weight, pad_left=obd_dim3, pad_right=obd_dim1)
    syn2_act_padded = zero_pad(syn2.actions.weight, pad_left=act_dim3, pad_right=act_dim1)

    if task2_path:
        syn3_obs_padded = zero_pad(syn3.observations.weight, pad_left=0, pad_right=merge_obd_dim - obd_dim3)
        syn3_act_padded = zero_pad(syn3.actions.weight, pad_left=0, pad_right=merge_act_dim - act_dim3)
        merge_syn.observations.weight.data = torch.cat((syn1_obs_padded, syn2_obs_padded, syn3_obs_padded), dim=0)
        merge_syn.actions.weight.data = torch.cat((syn1_act_padded, syn2_act_padded, syn3_act_padded), dim=0)
    else:
        merge_syn.observations.weight.data = torch.cat((syn1_obs_padded, syn2_obs_padded), dim=0)
        merge_syn.actions.weight.data = torch.cat((syn1_act_padded, syn2_act_padded), dim=0)

    return merge_syn


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

    # 测试的环境放在path1, path2和path3的顺序无所谓
    test_path = 'baseline/halfcheetah-medium-expert-v2/reward_halfcheetah-medium-expert-v2_4096.pt'
    task1_path = 'hopper-mr.pt'
    task2_path = 'walker2d-mr.pt'

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
    set_seed(config.seed, env)
    evaluator = Evaluator(env=env, config=config)

    return_list = []
    normalize_return_list = []

    if merged:
        synset = merge(test_path, task1_path, task2_path)
        return_list, normalize_return_list = evaluate_and_print(merged, synset, evaluator, env)
    else:
        if os.path.exists(test_path):
            synset = torch.load(test_path)
            return_list, normalize_return_list = evaluate_and_print(merged, synset, evaluator, env)

    print(np.mean(return_list))
    print(np.mean(normalize_return_list))


if __name__ == '__main__':
    sys.exit(main())

# 离线行为蒸馏算法的核心训练逻辑
import argparse
import copy
import importlib
import os
import sys
import time
import uuid
import warnings

import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
import torch
import torch.utils.data as tdata
import yaml
from torch import nn

import wandb
from configs.command_parser import command_parser, merge_args
from data_lib.data_bptt import SynSetBPTT
from data_load import D4RLDataset
from tools.evaluator import Evaluator
from utils import get_optimizer, set_seed, get_hms, wrap_env, compute_mean_std, normalize_states

# 忽略所有警告信息
warnings.filterwarnings('ignore')


# 定义简单的神经网络模块，用于表示一个可学习的标量参数
class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


# 将一个嵌套的字典转换为一个嵌套的命名空间对象
# 通过点号（.）访问字典中的键值对，而不需要使用字典的方括号（[]）语法
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# 将一个列表中的所有张量展平并拼接成一个一维张量
def flatten(data):
    return torch.cat([ele.flatten() for ele in data])


def wandb_init(config: dict):
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


"""
  Train function for compressor
"""


def train(config):
    # 定义在训练过程中进行评估的间隔点，每隔config.training.eval_every步进行一次评估
    eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)

    # 动态导入合成数据集库 data_lib.syndset
    synset_lib = importlib.import_module('data_lib.syndset')

    # 准备数据集
    env = gym.make(config.dataset.name)
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)
    # halfcheetah: 201798
    # antmaze: 998573

    # compute_mean_std 计算数据集中状态的均值和方差 (util)
    print(config.synset_size)
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    # normalize_states 数据归一化 (util)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    # wrap_env 将环境包装为归一化环境 (util)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # 创建数据加载器D4RLDataset (data_load)
    dataloader = tdata.DataLoader(D4RLDataset(dataset), batch_size=config.evaluation.batch_size, shuffle=True,
                                  num_workers=config.evaluation.num_workers)

    # set_seed 设置随机种子，确保实验的可重复性 (util)
    seed = config.seed
    set_seed(seed, env)

    # 初始化 WandB，记录实验数据
    print(config.group)
    print("init:", config.synset_init)
    wandb_init(config)

    # 定义离线策略模型的保存路径
    config.offline_policy_path = os.path.join(config.offline_policy_dir, 'Cal-QL-' + config.dataset.name,
                                              'checkpoint.pt')

    # 定义合成数据集 (data_lib/synset)
    # 使用 synset_lib.Net 创建合成数据集对象
    synset = synset_lib.Net(
        syn_dset_size=config.synset_size,
        observation_space=observation_space,
        action_space=action_space,
        config=config,
        device=config.device
    )

    # 调用 init_synset 方法初始化合成数据集对象
    synset.init_synset(config.synset_init, config.dataset.name, dataset)

    # 使用 SynSetBPTT创建BPTT模型 (data_lib/data_bptt)，用于训练合成数据集
    synset_bptt = SynSetBPTT(
        synset=synset,
        observation_space=observation_space,
        action_space=action_space,
        max_action=max_action,
        config=config
    ).to(config.device)

    # get_optimizer 定义合成数据集的优化器 (utils)
    synset_optimizer = get_optimizer(synset_bptt.synset.parameters(), config.synset_optim)

    # 创建评估器，用于评估合成数据集的性能 (tools/evaluator)
    evaluator = Evaluator(env=env, config=config)

    time_start = time.time()

    # 创建保存合成数据集的文件夹 (saved_synset)
    save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 训练过程迭代
    for it in range(config.training.n_iters):
        # 在评估间隔点评估模型性能
        if it in eval_intervals:
            # 到达评估间隔点，打印当前迭代次数和执行时间
            print("%s iterations" % (it))
            h, m, s = get_hms(time.time() - time_start)
            print("Execute time: %dh %dm %ds" % (h, m, s))

            # 深拷贝当前的合成数据集对象
            synset_copy = copy.deepcopy(synset_bptt.synset)

            # 保存合成数据集到指定路径
            if config.q_weight:
                save_path_name = os.path.join(
                    save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_'
                    + config.match_objective + '_q-weight_' + str(config.beta_weight) + '_init_'
                    + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')
            else:
                save_path_name = os.path.join(
                    save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_'
                    + config.match_objective + '_no-q-weight_init_' + str(config.synset_init) + '_iter_'
                    + str(it) + '_seed_' + str(config.seed) + '.pt')

            torch.save(synset_copy, save_path_name)

            # 使用评估器评估合成数据集的性能，记录归一化回报和离线损失 (tools/evaluator)
            trajectory_return_info = evaluator.trajectory_return(synset_copy)
            offline_loss = evaluator.offline_loss(synset_copy, dataloader)

            # 使用 WandB 记录评估结果
            print("Normalized return: " + str(trajectory_return_info["normalized_return"]))
            wandb.log(trajectory_return_info, step=it)
            wandb.log({"offline_test_loss": offline_loss}, step=it)

        # 用内循环优化compressor_bptt模型
        synset_optimizer.zero_grad()  # 清零梯度
        loss, dL_dc, dL_dw = synset_bptt.forward(test_dataloader=dataloader)  # 前向传播
        wandb.log({"train loss": loss}, step=it)  # 记录训练损失
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

        # 将计算得到的梯度分配给合成数据集的参数
        synset.assign_grads(grads=[flatten(ele) for ele in dL_dc])

        # 对模型参数的梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(synset_bptt.parameters(), max_norm=2)

        # 更新模型参数，清零优化器中的梯度
        synset_optimizer.step()
        synset_optimizer.zero_grad()

        # 归一化 actions
        # with torch.no_grad():
            # synset_bptt.synset.actions.weight.data = synset_bptt.synset._normalize_actions(
            # synset_bptt.synset.actions.weight.data)

            # synset_bptt.synset.actions.weight.data = torch.clamp(synset_bptt.synset.actions.weight.data, -1.0, 1.0)

        #     synset_bptt.synset.actions.weight.data = synset_bptt.synset._standardize_actions(
        #         synset_bptt.synset.actions.weight.data)
        #
        # print(synset_bptt.synset.actions.weight.data)

    print('Training completed.')


def parse_args_and_config():
    # parse arguments
    parser = command_parser()
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # merge arguments and config
    config = merge_args(args, config)

    return args, config


def main():
    # args and config
    args, config = parse_args_and_config()

    # setattr是Python的内置函数，用于设置对象的属性。
    # 作用：将指定的值赋给对象的指定属性。
    setattr(config, 'pid', str(os.getpid()))

    train(config)


if __name__ == '__main__':
    sys.exit(main())

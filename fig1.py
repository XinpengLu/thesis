import argparse
import os
import sys

import gym
import d4rl
import d4rl.gym_mujoco
import numpy as np
import torch
import yaml
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

from configs.command_parser import command_parser, merge_args
from network import TanhGaussianPolicy, FullyConnectedQFunction
from tools.evaluator import Evaluator
from utils import set_seed, normalize_states, wrap_env, compute_mean_std, eval_actor, set_env_seed


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def illu(config):
    env = gym.make(config.dataset.name)
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    max_action = float(env.action_space.high[0])

    def load_policy(policy_path):
        policy = TanhGaussianPolicy(observation_space[0], action_space[0], max_action, orthogonal_init=True)
        checkpoint = torch.load(policy_path)

        policy.load_state_dict(state_dict=checkpoint["actor"])
        policy.eval()
        return policy.to("cuda")

    def load_critic(critic_path):
        critic = FullyConnectedQFunction(observation_space[0], action_space[0], True, 5)
        checkpoint = torch.load(critic_path)
        critic.load_state_dict(state_dict=checkpoint["critic1"])
        critic.eval()
        return critic.to("cuda")

    def plot(policy, critic):
        actions_over_time = []
        q_values = []
        state, done = env.reset(), False
        trajectory = [state[:2]]

        for _ in range(200):
            action = policy.act(state, 'cuda')
            action = torch.tensor(action, dtype=torch.float32).to('cuda')
            state = torch.tensor(state, dtype=torch.float32).to('cuda')
            q_value = critic(state, action).detach()
            q_values.append(q_value.item())

            action = action.cpu().numpy()
            state = state.cpu().numpy()

            actions_over_time.append(action)
            state, _, done, _ = env.step(action)
            trajectory.append(state[:2])

        actions_over_time = np.array(actions_over_time)
        trajectory = np.array(trajectory)
        print("len:", len(trajectory))
        print("Q:", len(q_values))
        print(q_values)

        mean_action = np.mean(actions_over_time, axis=0)
        std_action = np.std(actions_over_time, axis=0)

        print(f"动作均值: {mean_action}")
        print(f"动作标准差: {std_action}")

        # 检查是否大多数动作接近 0
        threshold = 0.05  # 定义"接近零"的阈值
        percentage_near_zero = np.mean(np.abs(actions_over_time) < threshold) * 100
        print(f"接近零的动作比例: {percentage_near_zero:.2f}%")

        # 绘制直方图
        sns.histplot(actions_over_time.flatten(), bins=50, kde=True)
        plt.title("Distilled Data State Distribution")
        plt.show()


        plt.figure(figsize=(10, 6))
        plt.hist(actions_over_time.flatten(), bins=50, alpha=0.75, color='blue')
        plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2, label='Zero Action')
        plt.xlabel("Action Value")
        plt.ylabel("Frequency")
        plt.title("Action Distribution in AntMaze")
        plt.legend()
        plt.show()

        plt.plot(actions_over_time)
        plt.title("Action Values Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.show()

        # 可视化轨迹
        plt.figure(figsize=(6, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', alpha=0.3, linestyle='-', color='blue')
        plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='red', label='Start')
        plt.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], color='green', label='End')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("AntMaze Trajectory")
        plt.legend()
        plt.show()

    dataset = d4rl.qlearning_dataset(env)

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
    print(state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    seed = config.seed
    set_seed(seed, env)
    evaluator = Evaluator(env=env, config=config)
    save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])
    eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)
    it = eval_intervals[-2]

    if config.q_weight:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(
            config.synset_size) + '_' + config.match_objective + '_q-weight_' + str(
            config.beta_weight) + '_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(
            config.seed) + '.pt')
    else:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(
            config.synset_size) + '_' + config.match_objective + '_no-q-weight_init_' + str(
            config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')

    config.offline_policy_path = os.path.join(config.offline_policy_dir, 'Cal-QL-' + config.dataset.name,
                                              'checkpoint.pt')
    print("666:", config.offline_policy_path)
    save_path_name = 'antmaze-umaze_size_2048_offline_policy_q-weight_0.02_init_real_iter_49000_seed_0.pt'
    policy = load_policy(config.offline_policy_path)
    critic = load_critic(config.offline_policy_path)

    # trajectory_return_info = eval_actor(eval_env, policy, "cuda", 10, seed)
    # print(trajectory_return_info)

    synset = torch.load(save_path_name)
    obs_array = synset.observations.weight.data.cpu().numpy()
    act_array = synset.actions.weight.data.cpu().numpy()

    # sns.histplot(obs_array.flatten(), bins=50, kde=True)
    # plt.title("Distilled Data State Distribution")
    # plt.show()
    #
    # sns.histplot(act_array.flatten(), bins=50, kde=True)
    # plt.title("Distilled Data Action Distribution")
    # plt.show()

    _, policy_2 = evaluator.trajectory_return(synset)
    # print(trajectory_return_info)
    plot(policy, critic)
    plot(policy_2, critic)


def parse_args_and_config():
    parser = command_parser()
    args = parser.parse_args()
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    config = merge_args(args, config)
    return args, config


def main():
    args, config = parse_args_and_config()
    setattr(config, 'pid', str(os.getpid()))
    illu(config)


if __name__ == '__main__':
    sys.exit(main())

# 评估合成数据集
import os
import numpy as np
import yaml
import argparse
import gym
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from dataclasses import asdict

from torch import nn

from data_load import D4RLDataset
import d4rl.gym_mujoco

from configs.command_parser import command_parser, merge_args
from network import FullyConnectedQFunction, TanhGaussianPolicy
from utils import compute_mean_std, normalize_states, wrap_env
import warnings

warnings.filterwarnings('ignore')


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def load_critic(env, critic_path):
    critic = FullyConnectedQFunction(env.observation_space.shape[0],
                                     env.action_space.shape[0])
    checkpoint = torch.load(critic_path)
    critic.load_state_dict(state_dict=checkpoint["critic1"])
    critic.eval()
    return critic.to("cuda")

def load_policy(env,policy_path):
    policy = TanhGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0],
                                float(env.action_space.high[0]), orthogonal_init=True)
    checkpoint = torch.load(policy_path)
    policy.load_state_dict(state_dict=checkpoint["actor"])
    policy.eval()
    return policy.to('cuda')

env_id = 'halfcheetah-medium-replay-v2'
env = gym.make(env_id)
dataset = d4rl.qlearning_dataset(env)
state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)
save_path_name = 'halfcheetah-mr-0.006.pt'

synset = torch.load(save_path_name)
obs_array = synset.observations.weight.data.cpu().numpy()
act_array = synset.actions.weight.data.cpu().numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(obs_array.flatten(), bins=50, alpha=0.75)
plt.title("Distribution of Observations")
plt.subplot(1, 2, 2)
plt.hist(act_array.flatten(), bins=50, alpha=0.75)
plt.title("Distribution of Actions")
# plt.show()

print("**" * 20)
# 加载数据
obs_dataset = dataset["observations"]  # 原始状态
act_dataset = dataset["actions"]  # 原始动作

obs_synset = synset.observations.weight.data.cpu().numpy()  # 合成数据状态
act_synset = synset.actions.weight.data.cpu().numpy()  # 合成数据动作

# 计算均值和标准差
obs_mean_dataset, obs_std_dataset = np.mean(obs_dataset, axis=0), np.std(obs_dataset, axis=0)
act_mean_dataset, act_std_dataset = np.mean(act_dataset, axis=0), np.std(act_dataset, axis=0)

obs_mean_synset, obs_std_synset = np.mean(obs_synset, axis=0), np.std(obs_synset, axis=0)
act_mean_synset, act_std_synset = np.mean(act_synset, axis=0), np.std(act_synset, axis=0)

# 直方图比较
plt.figure(figsize=(12, 12))

plt.subplot(4, 2, 1)
plt.hist(obs_dataset.flatten(), bins=50, alpha=0.5, color='b', label="Original Dataset")
plt.title("State Distribution")
plt.legend()

plt.subplot(4, 2, 2)
plt.hist(obs_synset.flatten(), bins=50, alpha=0.5, color='r', label="Synthetic Dataset")
plt.title("State Distribution")
plt.legend()

plt.subplot(4, 2, 3)
plt.hist(act_dataset.flatten(), bins=50, alpha=0.5, color='b', label="Original Dataset")
plt.title("Action Distribution")
plt.legend()

plt.subplot(4, 2, 4)
plt.hist(act_synset.flatten(), bins=50, alpha=0.5, color='r', label="Synthetic Dataset")
plt.title("Action Distribution")
plt.legend()

# 统计对比
plt.subplot(4, 2, 5)
plt.bar(np.arange(len(obs_mean_dataset)), obs_mean_dataset, alpha=0.5, color='b', label="Original Dataset")
plt.title("State Mean Comparison")
plt.legend()

plt.subplot(4, 2, 6)
plt.bar(np.arange(len(obs_mean_synset)), obs_mean_synset, alpha=0.5, color='r', label="Synthetic Dataset")
plt.title("State Mean Comparison")
plt.legend()

plt.subplot(4, 2, 7)
plt.bar(np.arange(len(act_mean_dataset)), act_mean_dataset, alpha=0.5, color='b', label="Original Dataset")
plt.title("Action Mean Comparison")
plt.legend()

plt.subplot(4, 2, 8)
plt.bar(np.arange(len(act_mean_synset)), act_mean_synset, alpha=0.5, color='r', label="Synthetic Dataset")
plt.title("Action Mean Comparison")
plt.legend()

plt.tight_layout()
# plt.show()

syn_dset_size = 256
offline_policy_path = os.path.join('./offline_policy_checkpoints', 'Cal-QL-' + env_id,
                                   'checkpoint.pt')
offline_critic = load_critic(env, offline_policy_path)
q_value = offline_critic(torch.Tensor(dataset['observations']).to('cuda'),
                         torch.Tensor(dataset['actions']).to('cuda'))

# 选择Q值最高的样本
idx = torch.argsort(q_value.detach().cpu(), descending=True)
selected_idx = idx[:syn_dset_size]
selected_obs = dataset['observations'][selected_idx]
selected_actions = dataset['actions'][selected_idx]

# PCA降维可视化（仅取前两主成分）
pca = PCA(n_components=2)
obs_pca_dataset = pca.fit_transform(obs_dataset)
obs_pca_synset = pca.transform(obs_synset)
obs_pca_selected = pca.transform(selected_obs)

plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(obs_pca_dataset[:, 0], obs_pca_dataset[:, 1], alpha=0.5, label="Original Dataset", s=10)
plt.scatter(obs_pca_synset[:, 0], obs_pca_synset[:, 1], alpha=0.5, label="Synthetic Dataset", s=10)
plt.scatter(obs_pca_selected[:, 0], obs_pca_selected[:, 1], alpha=0.5, label="Selected Dataset", s=10)
plt.title("PCA Projection of States")
plt.legend()
plt.show()
# plt.savefig("states.pdf", format="pdf")



pca = PCA(n_components=2)

policy = load_policy(env, offline_policy_path)
act_opt, _ = policy(torch.tensor(obs_synset).to('cuda'))
act_opt = act_opt.cpu().detach().numpy()

act_pca_dataset = pca.fit_transform(act_dataset)
act_pca_synset = pca.transform(act_synset)
act_pca_selected = pca.transform(selected_actions)
act_pca_opt = pca.transform(act_opt)

plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(act_pca_dataset[:, 0], act_pca_dataset[:, 1], alpha=0.5, label="Original Dataset", s=10)
plt.scatter(act_pca_synset[:, 0], act_pca_synset[:, 1], alpha=0.5, label="Synthetic Dataset", s=10)
plt.scatter(act_pca_opt[:, 0], act_pca_opt[:, 1], alpha=0.5, label="Optimal Dataset", s=10)
plt.scatter(act_pca_selected[:, 0], act_pca_selected[:, 1], alpha=0.5, label="Selected Dataset", s=10)
plt.title("PCA Projection of Actions")
plt.legend()
plt.show()
# plt.savefig("actions.pdf", format="pdf")

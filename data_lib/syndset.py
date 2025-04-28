# 合成数据集初始化
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import FullyConnectedQFunction
from sklearn.cluster import KMeans, kmeans_plusplus

# 导入sklearn的聚类算法用于数据选择
# 定义了不同环境的初始状态和噪声缩放参数（hopper，halfcheetah，walker2d）
ENV_START_STATE = {"hopper": [1.25] + [0] * 10, "halfcheetah": [0] * 17, "walker2d": [1.25] + [0] * 16}
ENV_RESET_SCALE = {"hopper": 5e-3, "halfcheetach": 0.1, "walker2d": 5e-3}

''' Synthetic data generator '''


class Net(nn.Module):
    def __init__(
            self,
            syn_dset_size,
            observation_space,
            action_space,
            config,
            device
    ):
        super(Net, self).__init__()
        self.name = 'syn data'
        # 数据集规模 syn_dset_size
        self.syn_dset_size = syn_dset_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        self.prev_imgs = None

        # 使用nn.Embedding存储合成数据，observations和actions分别对应状态和动作
        # observations形状：(syn_dset_size, obs_dim)，actions形状：(syn_dset_size, act_dim)
        # 目的：将observation和action设计为可学习参数，可以通过梯度优化的方法调整数据，使得训练更高效
        self.observations = nn.Embedding(self.syn_dset_size, np.prod(self.observation_space))
        self.actions = nn.Embedding(self.syn_dset_size, np.prod(self.action_space))

        self.config = config

        # 通过load_critic加载预训练的Q函数网络，用于评估数据质量（checkpoint）
        self.offline_critic = self.load_critic(self.config.offline_policy_path)

    # 源代码中的critic
    def load_critic(self, critic_path):
        critic = FullyConnectedQFunction(self.observation_space[0], self.action_space[0], self.config.orthogonal_init,
                                         self.config.n_hidden_layers)
        checkpoint = torch.load(critic_path)

        critic.load_state_dict(state_dict=checkpoint["critic1"])
        critic.eval()

        return critic.to(self.device)

    # 数据初始化方法
    def init_synset(self, init_type, env, dataset):
        # 从真实数据集dataset中随机采样
        if init_type == 'real':
            perm = torch.randperm(dataset['observations'].shape[0])
            selected_idx = perm[:self.syn_dset_size]

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        # 选择Q值最高的样本
        elif init_type == 'q-value-real':
            q_value = self.offline_critic(torch.Tensor(dataset['observations']).to('cuda'),
                                          torch.Tensor(dataset['actions']).to('cuda'))
            print("q_value:", q_value)
            # 选择Q值最高的样本
            idx = torch.argsort(q_value.detach().cpu(), descending=True)
            selected_idx = idx[:self.syn_dset_size]

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        # 先Kmeans聚类，再在各簇中选q值最高的样本，平衡多样性和重量
        elif init_type == 'q-value-kmeans-real':
            model = KMeans(n_clusters=self.syn_dset_size, random_state=0, max_iter=1)
            kmeans = model.fit(dataset['observations'])
            cluster_label = kmeans.labels_

            q_value = self.offline_critic(torch.Tensor(dataset['observations']).to('cuda'),
                                          torch.Tensor(dataset['actions']).to('cuda'))
            q_min = torch.min(q_value.detach().cpu())
            idx = torch.argsort(q_value.detach().cpu(), descending=True)

            selected_idx = []
            for i in range(self.syn_dset_size):
                q_value_copy = copy.deepcopy(q_value.detach().cpu())
                q_value_copy[cluster_label != i] = q_min - 1.
                selected_idx.append(torch.argsort(q_value_copy.detach().cpu(), descending=True)[0])

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        # 用kmeans++选择代表性样本
        elif init_type == 'kmeans++':
            centers, selected_idx = kmeans_plusplus(dataset['observations'], n_clusters=self.syn_dset_size)

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        # 根据环境初始状态生成随机样本，动作在[-1,1]均匀分布
        elif init_type == 'random':
            env_class = env.split("-")[0]
            state_mean, state_std = torch.Tensor(ENV_START_STATE[env_class]), ENV_RESET_SCALE[env_class]

            self.observations.weight = torch.nn.Parameter(state_mean + state_std * torch.randn(self.observation_space))
            self.actions.weight = torch.nn.Parameter(2 * torch.rand(self.action_space) - 1.)  # actions in [-1, 1]

        # 参数使用Xavier初始化
        elif init_type == 'xavier':
            torch.nn.init.xavier_uniform(self.observations.weight)
            torch.nn.init.xavier_uniform(self.actions.weight)

        # 利用轨迹进行初始化
        elif init_type == 'trajectory':
            observations, actions, rewards, terminals = dataset["observations"], dataset["actions"], dataset["rewards"], \
            dataset["terminals"]

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
            num_clusters = min(self.syn_dset_size, len(trajectory_representations))
            kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0).fit(trajectory_representations)
            cluster_labels = kmeans.labels_

            # 5. 在每个聚类中选择最佳轨迹
            selected_trajectories = []
            for i in range(num_clusters):
                cluster_trajs = [traj for traj, label in zip(successful_trajectories, cluster_labels) if label == i]
                best_traj = max(cluster_trajs,
                                key=lambda traj: len(traj["observations"]))  # 可改成 np.sum(traj["rewards"])
                selected_trajectories.append(best_traj)

            # 6. 累积数据，直到 `syn_dset_size`
            selected_observations, selected_actions = [], []
            total_count = 0

            for traj in selected_trajectories:
                remain = min(self.syn_dset_size - total_count, len(traj["observations"]))
                selected_observations.extend(traj["observations"][:remain])
                selected_actions.extend(traj["actions"][:remain])
                total_count += remain
                if total_count >= self.syn_dset_size:
                    break  # 达到目标数据量

            self.observations.weight = torch.nn.Parameter(torch.Tensor(selected_observations))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(selected_actions))

        else:
            raise Exception("Synset initialization type can note be recognized.")

    def _normalize_actions(self, actions, eps=1e-6):
        G = actions.abs().sum(dim=1, keepdim=True) + eps  # 计算 G，确保非零
        return actions / G  # 归一化

    def _standardize_actions(self, actions):
        mean = actions.mean()
        std = actions.std()
        return (actions - mean) / std

    # 梯度管理：接收外部计算的梯度并应用到Embedding参数，用于优化合成数据
    def assign_grads(self, grads):
        obs_grads = grads[0]
        actions_grads = grads[1]

        self.observations.weight.grad = obs_grads.to(self.observations.weight.data.device).view(
            self.observations.weight.shape)
        self.actions.weight.grad = actions_grads.to(self.actions.weight.data.device).view(self.actions.weight.shape)

    # 前向传播：返回存储的合成数据（状态和动作）
    def forward(self, placeholder=None):
        observations = self.observations
        actions = self.actions
        return observations, actions

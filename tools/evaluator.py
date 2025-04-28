# Evaluator 评估器类，用于评估离线强化学习中的数据蒸馏方法
import sys
import torch
import torch.optim as optim

import numpy as np
import copy

from utils import eval, eval_ensemble_actor, eval_actor, get_optimizer, get_policy, eval_actor_merge
from network import TanhGaussianPolicy, DetContPolicy

policy_params = {
    "hidden_shapes": [400, 300],
    "append_hidden_shapes": [],
    "tanh_action": True
}

"""
  An Evaluator class:
    first, training the actor with synset
    1. evaluate by sample in the env 在环境中运行策略，评估策略的表现
    2. evaluate by MSE loss on the offline data 计算策略对离线数据的均方误差（MSE）
"""


# 训练策略（用合成数据 synset 进行监督学习）
# 在环境中评估策略（计算策略能否执行良好的动作）
# 支持集成策略（训练多个策略进行评估）
# 计算离线误差（用 MSE 衡量策略对离线数据的拟合程度）

class Evaluator(object):
    def __init__(
            self,
            env,
            config
    ):

        # configs
        self.config = config
        self.env = env  # 环境提供状态、动作的维度和动作范围
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        # 策略类型：Gaussian或deterministic
        self.policy_type = config.policy_type
        # 是否使用多个策略集成以及集成策略的数量
        self.eval_ensemble = config.eval_ensemble
        self.ensemble_policy_num = config.ensemble_policy_num

        self.device = config.device
        # 损失函数：使用均方误差计算动作的拟合误差
        self.loss_func = torch.nn.MSELoss()

    def _create_policy(self, policy_type):
        # 创建策略（NN）：高斯策略或确定性策略
        if policy_type == 'gaussian':
            policy = TanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif policy_type == 'deterministic':
            policy = get_policy(
                input_shape=self.state_dim,
                output_shape=self.action_dim,
                policy_cls=DetContPolicy,
                policy_params=policy_params
            )
        else:
            raise Exception("Non-recognized policy_type.")

        return policy.to(self.device)

    def _get_pred_acts(self, policy_type, policy, obs):
        # 计算策略预测的动作：输入状态obs，输出动作pre_actions
        if policy_type == 'gaussian':
            pred_actions, _ = policy(obs)
        elif policy_type == 'deterministic':
            pred_actions = policy(obs)
        else:
            raise Exception("Non-recognized policy_type.")

        return pred_actions

    def _eval_policy(self, policy_type, policy):
        # 评估策略在环境中的表现：eval_actor和eval在utils中
        if policy_type == 'gaussian':
            eval_info = eval_actor(self.env, policy, device=self.config.device,
                                   n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        elif policy_type == 'deterministic':
            eval_info = eval(self.env, policy, device=self.config.device,
                             eval_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        else:
            raise Exception("Non-recognized policy_type.")

        return eval_info

    def _train(self, policy, synset):
        # 利用合成数据用监督学习的方法训练策略
        policy.train()
        self.config.bptt.inner_steps=1000
        for epoch in range(self.config.bptt.inner_steps):
            #obs, actions = synset.to(self.device)
            # 从synset中获取数据（合成数据）
            obs = synset.observations.weight.to(self.device)
            actions = synset.actions.weight.to(self.device)

            # 定义优化器，get_optimizer在utils
            optimizer = get_optimizer(policy.parameters(), self.config.bptt_optim)

            # 计算策略预测的动作
            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)
            policy_loss = self.loss_func(pred_actions, actions)

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        policy.eval()
        return None

    def _create_and_train_ensemble_policy(self, policy_type, ensemble_policy_num, synset):
        # 训练多个策略（集成策略）
        policy_list = []
        for i in range(ensemble_policy_num):
            policy = self._create_policy(policy_type)
            self._train(policy, synset)
            policy_list.append(policy)

        return policy_list

    # 通过环境评估策略
    def trajectory_return(self, synset):
        # 集成策略评估，使用多策略进行评估，eval_ensemble_actor在utils
        if self.eval_ensemble:
            policy_list = self._create_and_train_ensemble_policy(self.policy_type, self.ensemble_policy_num, synset)
            eval_info = eval_ensemble_actor(self.env, policy_list, device=self.config.device,
                                            n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        # 单个策略评估，调用上面的_eval_policy
        else:
            policy = self._create_policy(self.policy_type)
            self._train(policy, synset)
            eval_info = self._eval_policy(self.policy_type, policy)

        return eval_info

        # return eval_info, policy

    def _train_merge(self, policy, synset):
        policy.train()
        for epoch in range(1000):
            obs = synset.observations.weight.to(self.device)
            actions = synset.actions.weight.to(self.device)
            optimizer = get_optimizer(policy.parameters(), self.config.bptt_optim)
            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)
            policy_loss = self.loss_func(pred_actions, actions)
            # print(policy_loss.item())
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        policy.eval()
        return None

    def _create_policy_merge(self, state_dim, action_dim):
        policy = TanhGaussianPolicy(state_dim, action_dim, self.max_action, orthogonal_init=True)
        return policy.to(self.device)

    def trajectory_return_merge(self, synset):
        state_dim = synset.observations.weight.shape[1]
        action_dim = synset.actions.weight.shape[1]
        policy = self._create_policy_merge(state_dim, action_dim)
        self._train_merge(policy, synset)
        eval_info = eval_actor_merge(self.env, policy, device=self.config.device, synset=synset,
                               n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        return eval_info

    # 计算策略在离线数据上的误差
    def offline_loss(self, synset, offline_testloader):
        # 创建策略并训练
        policy = self._create_policy(self.policy_type)
        trained_policy = self._train(policy, synset)

        loss_list = []
        for i, batch in enumerate(offline_testloader):
            obs = batch['obs']
            actions = batch['acts']

            obs = torch.Tensor(obs).to(self.device)
            actions = torch.Tensor(actions).to(self.device)

            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)

            loss = self.loss_func(pred_actions, actions)
            loss_list.append(loss.item())

        return np.mean(loss_list)

# Based on https://github.com/princetonvisualai/RememberThePast-DatasetDistillation
# 实现了BPTT（Backpropagation Through Time）算法，用于训练合成数据集。
import copy
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import torch.nn.functional as F
from gym.spaces import Box

from utils import get_optimizer, get_policy
from network import TanhGaussianPolicy, DetContPolicy, FullyConnectedQFunction

policy_params = {
    "hidden_shapes": [400, 300],
    "append_hidden_shapes": [],
    "tanh_action": True
}


class SynSetBPTT(nn.Module):
    def __init__(
            self,
            synset,
            observation_space,
            action_space,
            max_action,
            config
    ):
        super(SynSetBPTT, self).__init__()
        self.synset = synset  # 合成数据集
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_action = max_action

        self.policy_type = config.policy_type

        # 损失函数使用行为克隆损失
        self.loss_func = self.bc_loss
        self.device = config.device

        # 离线RL策略和离线Q网络
        self.offline_policy = self.load_policy(self.config.offline_policy_path)
        self.offline_critic = self.load_critic(self.config.offline_policy_path)

    # 创建策略
    def _create_policy(self, policy_type):
        if policy_type == 'gaussian':
            policy = TanhGaussianPolicy(self.observation_space[0], self.action_space[0], self.max_action,
                                        orthogonal_init=True)
        elif policy_type == 'deterministic':
            policy = get_policy(
                input_shape=self.observation_space,
                output_shape=self.action_space[0],
                policy_cls=DetContPolicy,
                policy_params=policy_params
            )
        else:
            raise Exception("Non-recognized policy_type.")

        return policy.to(self.device)

    # 返回策略预测动作
    def _get_pred_acts(self, policy_type, policy, obs):
        if policy_type == 'gaussian':
            pred_actions, _ = policy(obs)
        elif policy_type == 'deterministic':
            pred_actions = policy(obs)
        else:
            raise Exception("Non-recognized policy_type.")

        return pred_actions

    def _get_pred(self, policy_type, policy, obs):
        if policy_type == 'gaussian':
            pred_actions, logp_action = policy(obs)
        else:
            raise Exception("Non-recognized policy_type.")
        return pred_actions, logp_action

    """
      Summation
    """

    def sum(self, inputs):
        output = 0
        for ele in inputs:
            output += ele
        return output

    """
      Flatten tensors
    """

    def flatten(self, data):
        return torch.cat([ele.flatten() for ele in data])

    """
      BC loss
    """

    # backbone就是policy
    def bc_loss(self, backbone, obs, actions):
        mse_loss = torch.nn.MSELoss()
        pred_actions = self._get_pred_acts(self.policy_type, backbone, obs)
        policy_loss = mse_loss(pred_actions, actions)
        return policy_loss

    """
      Weighted MSE loss
    """

    # 加权MSE损失，对损失施加权重 weight，使得高 Q-value 样本影响更大
    def weighted_mse_loss(self, backbone, obs, actions, weight, alpha):
        pred_actions, log_pi = self._get_pred(self.policy_type, backbone, obs)
        # pred_actions = self._get_pred_acts(self.policy_type, backbone, obs)

        weight_mat = weight.expand(self.action_space[0], -1).t()
        policy_loss = torch.mean(((pred_actions - actions) ** 2) * weight_mat)
        entropy = -torch.mean(log_pi)
        # print("alpha:", alpha)
        # print("policy_loss:", policy_loss)
        # print("entropy:", entropy * alpha)

        # return torch.mean(policy_loss)
        return torch.mean(policy_loss + alpha * entropy)

    """
      Load well-trained policy from offline RL
    """

    # 加载离线策略 & Q 网络
    def load_policy(self, policy_path):
        policy = self._create_policy(self.policy_type)
        checkpoint = torch.load(policy_path)

        policy.load_state_dict(state_dict=checkpoint["actor"])
        policy.eval()

        return policy.to(self.device)

    def load_critic(self, critic_path):
        critic = FullyConnectedQFunction(self.observation_space[0], self.action_space[0], self.config.orthogonal_init,
                                         self.config.n_hidden_layers)
        checkpoint = torch.load(critic_path)

        critic.load_state_dict(state_dict=checkpoint["critic1"])
        critic.eval()

        return critic.to(self.device)

    """
      forward with inner loops, based on addressing type
    """

    # 计算训练目标（内循环）：输入为测试数据集，输出为损失函数和梯度
    def forward(self, test_dataloader):
        # 初始化在合成数据集上训练的策略
        backbone = self._create_policy(self.policy_type)
        # 策略优化器
        backbone_opt = get_optimizer(backbone.parameters(), self.config.bptt_optim)

        # loss：损失值
        # dL_dc：损失对 synset（合成数据）的梯度
        # dL_dw：损失对策略网络参数 w 的梯度
        loss, dL_dc, dL_dw = self.inner_loop(
            backbone,
            backbone_opt,
            test_dataloader,
        )
        return loss, dL_dc, dL_dw


    """
      inner loop with label-based addressing (standard dataset distillation)
    """

    def inner_loop(self, backbone, backbone_opt, test_dataloader):
        # storing gradients and weights offline, allow reversible
        backbone.zero_grad()
        self.synset.zero_grad()
        # gws, ws, datums, backbone_trained
        ws, backbone_trained = self.bptt_efficient_forward(
            backbone,
            backbone_opt,
            self.config.bptt.inner_steps,
            self.loss_func,
        )

        # 计算目标损失函数H（OBC和PBC两种, 对应match_objective参数）
        if self.config.match_objective == 'offline_data':
            loss = self.batched_test_loss(backbone_trained, test_dataloader, self.config.bptt.test_batch_num)
        elif self.config.match_objective == 'offline_policy':
            loss = self.objective_loss(backbone, self.offline_policy, self.offline_critic, test_dataloader,
                                       self.config.bptt.test_batch_num)
        else:
            raise Exception("Non-recognized match_objective.")

        self.synset.zero_grad()
        dL_dw = torch.autograd.grad(loss, list(backbone_trained.parameters()))

        # 反向传播计算损失对合成数据集和参数的梯度
        dL_dw, dL_dc = self.bptt_efficient_backward(
            ws,
            dL_dw,
            self.config.bptt.inner_steps,
            lr=self.config.bptt_optim.lr,
            momentum=self.config.bptt_optim.momentum,
            loss_func=self.loss_func,
            device=self.config.device
        )

        return loss, dL_dc, dL_dw

    """
      Test loss of well-trained loss on real data
    """

    # 计算行为克隆BC的损失（OBD的损失函数）
    def batched_test_loss(self, backbone, test_dataloader, test_batch_num):
        loss = 0
        for i, batch in enumerate(test_dataloader):
            if i == test_batch_num:
                break
            obs = batch['obs']
            actions = batch['acts']

            obs = torch.Tensor(obs).to(self.device)
            actions = torch.Tensor(actions).to(self.device)

            loss += self.loss_func(backbone, obs.detach(), actions.detach())

        return loss / test_batch_num

    """
       Objective loss: measure the diff between policy and Offline RL policy
    """

    def objective_loss(self, backbone, offline_policy, offline_critic, test_dataloader, test_batch_num):
        loss = 0
        for i, batch in enumerate(test_dataloader):
            if i == test_batch_num:
                break
            obs = batch['obs']
            obs = torch.Tensor(obs).to(self.device)

            # 获取最优的动作
            pred_optim_actions = self._get_pred_acts(self.policy_type, offline_policy, obs)
            q_value = self.offline_critic(obs, pred_optim_actions).detach()
            # print("q_value_shape:", q_value.shape)

            # V_val = q_value.mean(dim=-1, keepdim=True)
            # print("V_val:", V_val)
            # advantage = q_value - V_val
            # advantage_weight = self.config.beta_weight * advantage

            q_weight = self.config.beta_weight * q_value

            # 是否采用Q加权（PBC和Av-PBC）
            if self.config.q_weight:
                loss += self.weighted_mse_loss(backbone, obs.detach(), pred_optim_actions.detach(), q_weight, self.config.alpha)

                # Av-PBC的损失函数
                # loss += self.weighted_mse_loss(backbone, obs.detach(), pred_optim_actions.detach(), q_weight)
            else:
                # PBC的损失函数
                loss += self.loss_func(backbone, obs.detach(), pred_optim_actions.detach())

        return loss / test_batch_num

    """
      BPTT forward pass. Weights stored on cpus (hack).
    """

    # 内循环中基于BC在合成数据集上训练初始化策略
    def bptt_efficient_forward(
            self,
            backbone,
            backbone_opt,
            inner_steps,
            loss_func
    ):
        # backbone.load_state_dict(self.offline_policy.state_dict())
        backbone.train()
        ws = []
        datums = []
        for idx in range(inner_steps):
            obs = self.synset.observations.weight.to(self.device)
            actions = self.synset.actions.weight.to(self.device)
            # 行为克隆损失函数loss (l_BC)
            loss = loss_func(backbone, obs, actions)
            backbone_opt.zero_grad()
            # 保存梯度下降前的backbone
            ws.append(copy.deepcopy(backbone).cpu())
            loss.backward()
            backbone_opt.step()
        return ws, backbone

    """
      Backward computation of gradients
      Return:
      - dL_dw: grads wrt model weights
      - dL_dc: grads wrt compressors
    """

    def bptt_efficient_backward(
            self,
            ws,
            dL_dw,
            inner_steps,
            lr,
            momentum,
            loss_func,
            device
    ):

        obs = self.synset.observations.weight.to(self.device)
        actions = self.synset.actions.weight.to(self.device)

        obs.requires_grad_()
        actions.requires_grad_()

        gobs = []
        gactions = []
        gindices = []

        dL_dv = [0] * len(dL_dw)
        for backbone_w in reversed(list(ws)):
            dgw = [lr * ele.neg() for ele in dL_dw]  # gw is already weighted by lr, so simple negation
            dL_dv = [dL_dv_ele + dgw_ele for dL_dv_ele, dgw_ele in zip(dL_dv, dgw)]

            backbone_w.to(device)
            backbone_w.zero_grad()

            hvp_in = [obs, actions]
            loss = loss_func(backbone_w, obs, actions)

            params = list(backbone_w.parameters())
            hvp_in.extend(params)
            gw = torch.autograd.grad(loss, params, create_graph=True)

            hvp_grad = torch.autograd.grad(
                outputs=(self.flatten(gw),),
                inputs=hvp_in,
                grad_outputs=(self.flatten(dL_dv),),
            )

            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                if len(gobs) == 0:
                    gobs = hvp_grad[0]
                    gactions = hvp_grad[1]
                else:
                    gobs += hvp_grad[0]
                    gactions += hvp_grad[1]

                indent = 2  # because the first g is gobs, the second g is gactions, then set indent=2

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                for idx in range(len(dL_dw)):
                    dL_dw[idx].add_(hvp_grad[idx + indent])

            dL_dv = [dL_dv_ele * momentum for dL_dv_ele in dL_dv]

        dL_dc = (gobs, gactions)

        return dL_dw, dL_dc

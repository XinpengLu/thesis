# 网络架构
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from typing import Any, Dict, List, Optional, Tuple, Union


# 使用Fan-in初始化方法初始化权重
def _fanin_init(tensor, alpha=0):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    # bound = 1. / np.sqrt(fan_in)
    bound = np.sqrt(1. / ((1 + alpha * alpha) * fan_in))
    return tensor.data.uniform_(-bound, bound)


# 使用均匀分布初始化权重
def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)


# 使用常数值初始化偏置项
def _constant_bias_init(tensor, constant=0.1):
    tensor.data.fill_(constant)


# 初始化神经网络层的权重和偏置项
def layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init):
    weight_init(layer.weight)
    bias_init(layer.bias)


# 使用fanin_init初始化权重，使用constant_bias_init初始化偏置项
def basic_init(layer):
    layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)


# 使用_uniform_init初始化权重和偏置项
def uniform_init(layer):
    layer_init(layer, weight_init=_uniform_init, bias_init=_uniform_init)


# 使用正交初始化方法初始化权重，偏置项初始化为0
def _orthogonal_init(tensor, gain=np.sqrt(2)):
    nn.init.orthogonal_(tensor, gain=gain)


def orthogonal_init(layer, scale=np.sqrt(2), constant=0):
    layer_init(layer,
               weight_init=lambda x: _orthogonal_init(x, gain=scale),
               bias_init=lambda x: _constant_bias_init(x, 0))


# 全连接层MLP
class MLPBase(nn.Module):
    def __init__(
            self, input_shape, hidden_shapes,
            activation_func=nn.ReLU,
            init_func=basic_init,
            last_activation_func=None):
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []

        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for next_shape in hidden_shapes:
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            self.fcs.append(activation_func())
            input_shape = next_shape
            self.output_shape = next_shape

        self.fcs.pop(-1)
        self.fcs.append(self.last_activation_func())
        self.seq_fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        return self.seq_fcs(x)


class Net(nn.Module):
    def __init__(
            self,
            output_shape,
            base_type,  # 基础网络类型，如MLPBase
            append_hidden_shapes=[],  # 附加隐藏层
            append_hidden_init_func=basic_init,
            net_last_init_func=uniform_init,
            activation_func=nn.ReLU,
            **kwargs):
        super().__init__()
        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for next_shape in append_hidden_shapes:
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            self.append_fcs.append(self.activation_func())
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

        self.append_fcs.append(self.last)
        self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    def forward(self, x):
        out = self.base(x)
        out = self.seq_append_fcs(out)
        return out


# 确定性连续策略网络：继承Net类
class DetContPolicy(Net):
    def __init__(self, tanh_action=False, **kwargs):
        # 是否使用tanh函数对动作进行缩放
        self.tanh_action = tanh_action
        super(DetContPolicy, self).__init__(**kwargs)

    # 前向传播方法，输入状态x，返回策略网络的输出
    def forward(self, x):
        if self.tanh_action:
            return torch.tanh(super().forward(x))
        return super().forward(x)

    # 在评估模式下，输入状态x，返回策略网络的输出并转换为NumPy数组
    def eval_act(self, x):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    # 探索模式下，输入状态x，返回策略网络的输出
    def explore(self, x):
        return {
            "action": self.forward(x).squeeze(0)
        }


# 重参数化的Tanh高斯分布: 计算给定样本sample在分布中的对数概率
class ReparameterizedTanhGaussian(nn.Module):
    # log_prob: https://pytorch.org/docs/stable/distributions.html#transformeddistribution
    def __init__(self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    # 计算给定样本的对数概率
    def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            # Normal 标准高斯分布：提供采样和概率密度计算的基本功能
            action_distribution = Normal(mean, std)
        else:
            # TransformedDistribution 在基础分布上施加非线性变换（如 tanh）
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        # 逐元素计算样本 sample 的对数概率
        # log_prob 适用于 PyTorch 的概率分布类，用来计算样本在给定分布下的对数概率
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    # 前向传播方法，输入均值mean和对数标准差log_std，返回采样的动作和对数概率
    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # log_std 对数标准差
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


# Tanh高斯策略网络
class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            # 调整输出对数标准差的缩放和偏移，用于控制动作分布的宽度
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            # 是否使用正交初始化权重
            orthogonal_init: bool = False,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        # 初始化参数网络
        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        # 控制 log_std 的动态缩放和偏移的可学习参数
        # 模型可以对标准差进行更细粒度的调整，从而在训练过程中能够更好地适应不同的任务和环境
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)

        # 实现 Tanh 高斯分布的类，用于生成样本和计算对数概率
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    # 计算给定动作在 Tanh 高斯分布下的对数概率
    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor):
        # 如果动作是三维的（批量数据），重复观察值以匹配动作的形状
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)  # (batch_size, 2 * action_dim)

        # 将 base_network_output 张量沿着最后一个维度（即 dim=-1）分成两个部分，每部分的大小为 self.action_dim
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)

        # log_std 通过 log_std_multiplier 和 log_std_offset 调整
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()

        # 使用高斯分布的均值和标准差，结合 Tanh 变换，计算每个动作的对数概率
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,  # default False
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果 repeat 参数非空，扩展观察值以支持多个重复采样
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    # 这是一个推理方法，给定一个状态，生成相应的动作
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()
    # 推理过程中加上 @torch.no_grad() 装饰器，确保不计算梯度，从而提高效率和减少内存消耗


class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

        # 构造成(state, action)二元组，计算状态-动作值
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


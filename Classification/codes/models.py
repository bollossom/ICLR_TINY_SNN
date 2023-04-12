import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate
# from spikingjelly.clock_driven import functional, layer, surrogate, accelerating
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms
import math

from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn

import math
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None
import torch
import torch.nn as nn
import copy

class MemoryModule(nn.Module):
    # code from https://github.com/fangwei123456/spikingjelly
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _memoriesModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _memoriesModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _memoriesModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _memoriesModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a spiking neuron). The reset value of this variable will be ``value``.

        """
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _memoriesModule.reset-cn:

        重置所有有状态变量。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _memoriesModule.reset-en:

        Reset all stateful variables.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _memoriesModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _memoriesModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)

        for key, value in self._memories_rv.items():
            if isinstance(value, torch.Tensor):
                self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class BaseNode(MemoryModule):
    def __init__(self, init_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        # code from https://github.com/fangwei123456/spikingjelly
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = nn.Parameter(torch.tensor(init_v_threshold, dtype=torch.float))
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike
class ALIF(BaseNode):
    def __init__(self, init_tau=2.0, init_v_threshold=0.5, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False):
        super().__init__(init_v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) * self.w.sigmoid()
        else:

            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        return self.spiking()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'init_v_threshold={self.init_v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, use_plif, alpha_learnable, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        layer.Dropout(dpp, dropout_spikes=True),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
    )


class StaticNetBase(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.static_conv = None
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        for t in range(1, self.T):
            out_spikes_counter += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)

        return out_spikes_counter

class MNISTNet(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv = nn.Sequential(
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            layer.Dropout(0.5, dropout_spikes=True),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )

class FashionMNISTNet(MNISTNet):
    pass  # 与MNISTNet的结构完全一致

class Cifar10Net(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv = nn.Sequential(
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            ALIF(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )
def get_transforms(dataset_name):
    transform_train = None
    transform_test = None
    if dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])
    elif dataset_name == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train, transform_test

class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter

class NMNISTNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 34
        h = 34  # 原始数据集尺寸
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >>number_layer, dpp=0.5, class_num=10, init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class CIFAR10DVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, channels, number_layer, detach_reset):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)

class ASLDVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        # input size 256 * 256
        w = 256
        h = 256

        self.conv = nn.Sequential(
            Interpolate(size=256, mode='bilinear'),
        )

class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                             init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
from typing import Optional

import torch
import numpy as np


class NnModuleUtils(object):

    @classmethod
    def padding_calculate(cls, kernel_size, dilation=1):
        if kernel_size % 2 != 1:
            raise RuntimeError(f'kernel_size({kernel_size}) % 2 != 0')
        if kernel_size <= 0 or dilation <= 0:
            raise RuntimeError(f'kernel_size({kernel_size}) and dilation({dilation}) must be positive')
        result = round(kernel_size // 2)
        result *= dilation
        return result

    @classmethod
    def get_neuron_counts(cls, in_size, hidden_neurons, out_size):
        neuron_counts = []
        neuron_counts.append(in_size)
        neuron_counts.extend(hidden_neurons)
        neuron_counts.append(out_size)
        return neuron_counts

    @classmethod
    def from_numpy(cls, np_array, dtype=np.float32, cuda=None):
        t = torch.from_numpy(np_array.astype(dtype))
        t = t if cuda is None else NnModuleUtils.to_cuda(t, cuda=cuda)
        return t

    @classmethod
    def from_array(cls, arr, dtype=np.float32, cuda=None):
        if isinstance(arr, torch.Tensor):
            return cls.to_cuda(arr, cuda=cuda)
        t = NnModuleUtils.from_numpy(np.asarray(arr), dtype=dtype)
        t = t if cuda is None else NnModuleUtils.to_cuda(t, cuda=cuda)
        return t

    @classmethod
    def from_nontensor(cls, val, dtype=np.float32, cuda: Optional[bool]=None):
        if isinstance(val, torch.Tensor):
            return cls.to_cuda(val, cuda=cuda)
        if isinstance(val, list) or isinstance(val, np.ndarray):
            return NnModuleUtils.from_array(arr=val, dtype=dtype, cuda=cuda)
        t = torch.tensor(val)
        t = t if cuda is None else NnModuleUtils.to_cuda(t, cuda=cuda)
        return t

    @classmethod
    def cuda_is_available(cls, cuda=True):
        return torch.cuda.is_available() and cuda

    @classmethod
    def move_to(cls, t: torch.Tensor, device=None, cuda: bool=None):
        if (device is None) and (cuda is None):
            return t
        if device is not None:
            return NnModuleUtils.to_device(t, device=device)
        if cuda is not None:
            return NnModuleUtils.to_cuda(t, cuda=cuda)

    @classmethod
    def to_device(cls, t: torch.Tensor, device):
        if (not isinstance(t, list)) and (not isinstance(t, tuple)):
            return t.to(device=device)
        if not isinstance(device, list):
            device = [device for _ in range(len(t))]
        if len(device) != len(t):
            raise RuntimeError(f'len(device)({len(device)}) != len(t)({len(t)})')
        res = [cls.to_device(t[i], device[i]) for i in range(len(t))]
        return res if not isinstance(t, tuple) else tuple(res)

    @classmethod
    def to_cuda(cls, t: torch.Tensor, cuda=True):
        if cuda is None:
            return t
        if (not isinstance(t, list)) and (not isinstance(t, tuple)):
            return t.cuda() if cls.cuda_is_available(cuda=cuda) else t
        if not isinstance(cuda, list):
            cuda = [cuda for _ in range(len(t))]
        if len(cuda) != len(t):
            raise RuntimeError(f'len(cuda)({len(cuda)}) != len(t)({len(t)})')
        res = [cls.to_cuda(t[i], cuda[i]) for i in range(len(t))]
        return res if not isinstance(t, tuple) else tuple(res)

    @classmethod
    def unsqueeze_tensors(cls, tensors, dim=0):
        res = []
        for t in tensors:
            new_t = t.unsqueeze(dim)
            res.append(new_t)
        if isinstance(tensors, tuple):
            res = tuple(res)
        return res

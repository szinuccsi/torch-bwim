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
    def from_numpy(cls, np_array, dtype=np.float32):
        return torch.from_numpy(np_array.astype(dtype))

    @classmethod
    def from_array(cls, arr, dtype=np.float32):
        return NnModuleUtils.from_numpy(np.asarray(arr), dtype=dtype)

    @classmethod
    def from_nontensor(cls, val, dtype=np.float32):
        if isinstance(val, list) or isinstance(val, np.ndarray):
            return NnModuleUtils.from_array(arr=val, dtype=dtype)
        return torch.tensor(val)

    @classmethod
    def get_neuron_counts(cls, in_size, hidden_neurons, out_size):
        neuron_counts = []
        neuron_counts.append(in_size)
        neuron_counts.extend(hidden_neurons)
        neuron_counts.append(out_size)
        return neuron_counts

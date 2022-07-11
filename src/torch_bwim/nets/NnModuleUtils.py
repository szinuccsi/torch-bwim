import torch
import numpy as np


class NnModuleUtils(object):

    @classmethod
    def padding_calculate(cls, kernel_size, dilation=1):
        result = round(kernel_size // 2)
        result *= dilation
        return result

    @classmethod
    def from_numpy(cls, np_array):
        return torch.from_numpy(np_array.astype(np.float32))

    @classmethod
    def from_array(cls, arr):
        return NnModuleUtils.from_numpy(np.asarray(arr))

    @classmethod
    def get_neuron_counts(cls, in_size, hidden_neurons, out_size):
        neuron_counts = []
        neuron_counts.append(in_size)
        neuron_counts.extend(hidden_neurons)
        neuron_counts.append(out_size)
        return neuron_counts

import torch
import torch.nn as nn

from torch_bwim.NetBase import NetBase
from torch_bwim.modules.ActivationFunctions import ActivationFunctions
from torch_bwim.modules.NnModuleUtils import NnModuleUtils


class MultiLayerNet(NetBase):

    class Config(object):
        def __init__(self, in_size: int, out_size: int, hidden_neurons: list, activation_function: str=None,
                     dropout_p=0.0, batch_norm=False):
            super().__init__()
            self.in_size = in_size
            self.out_size = out_size
            self.hidden_neurons = hidden_neurons
            self.activation_function = activation_function
            self.dropout_p = dropout_p
            self.batch_norm = batch_norm

    def __init__(self, config: Config, random_state=None, activation_function=None):
        super().__init__(config=config, random_state=random_state)
        self.config = config
        cfg = config

        if activation_function is None:
            self.activation_function = ActivationFunctions.create_nonlinearity(cfg.activation_function)
        else:
            self.activation_function = activation_function
        self.linear_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        sizes = NnModuleUtils.get_neuron_counts(in_size=cfg.in_size, hidden_neurons=cfg.hidden_neurons,
                                                out_size=cfg.out_size)
        for i in range(1, len(sizes)):
            self.linear_layers.append(nn.Linear(sizes[i - 1], sizes[i], bias=self.bias))
            if cfg.batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(sizes[i]))
            else:
                self.batch_norm_layers.append(nn.Identity())
        if len(self.linear_layers) != len(self.batch_norm_layers):
            raise RuntimeError('linearLayers - batchNormLayers not same number of layers: {lin} - {bNorm}'.format(
                lin=len(self.linear_layers), bNorm=len(self.batch_norm_layers)
            ))
        self.dropout_layer = nn.Dropout(p=cfg.dropout_p)

    def _get_bias(self): return not self.config.batch_norm
    bias = property(_get_bias)

    def forward(self, t: torch.Tensor):
        for i in range(len(self.linear_layers)):
            linear_layer = self.linear_layers[i]
            b_norm_layer = self.batch_norm_layers[i]
            t = linear_layer(t)
            t = self.nonlinearity(t)
            t = b_norm_layer(t)
            t = self.dropout_layer(t)
        return t

    def __call__(self, t: torch.Tensor):
        return self.forward(t)

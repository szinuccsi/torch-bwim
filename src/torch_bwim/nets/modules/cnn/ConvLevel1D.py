from typing import Optional

import torch
import torch.nn as nn

from torch_bwim.nets.NetBase import NetBase
from torch_bwim.nets.modules.cnn.ConvBlock1D import ConvBlock1D


class ConvLevel1D(NetBase):

    class Config(NetBase.Config):

        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, layer_num: int,
                     activation_function: str, dropout_p: float = 0., residual: bool = True,
                     batch_norm: bool = False, bias: bool = True,
                     mode: str = 'same', padding_mode: str = 'replicate'):
            super().__init__()
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.kernel_size = kernel_size
            self.layer_num = layer_num
            self.activation_function = activation_function
            self.dropout_p = dropout_p
            self.residual = residual
            self.batch_norm = batch_norm
            self.bias = bias
            self.mode = mode
            self.padding_mode = padding_mode

    def __init__(self,  config: Config, random_state: Optional[int] = None,
                 activation_function: Optional[nn.Module] = None):
        super().__init__(config=config, random_state=random_state)
        self.config = config
        cfg = config

        self.upsampling_layer = self._conv_block_create(in_ch=cfg.in_ch, out_ch=cfg.out_ch,
                                                        activation_function=activation_function)
        layers = []
        for _ in range(cfg.layer_num):
            l = self._conv_block_create(
                in_ch=cfg.out_ch, out_ch=cfg.out_ch,
                activation_function=activation_function
            )
            layers.append(l)
        self.sequential_conv_blocks = nn.Sequential(*layers)

    def _conv_block_create(self, in_ch: int, out_ch: int, activation_function: Optional[nn.Module] = None):
        cfg = self.config
        return ConvBlock1D(
            in_ch=in_ch, out_ch=out_ch, kernel_size=cfg.kernel_size,
            activation_function=activation_function if activation_function is not None else cfg.activation_function,
            batch_norm=cfg.batch_norm, dropout_p=cfg.dropout_p, bias=cfg.bias, mode=cfg.mode,
            padding_mode=cfg.padding_mode
        )

    '''
        input: shape(batch; in_ch; len)
        return: shape(batch; out_ch; len)
    '''
    def forward(self, input: torch.Tensor):
        upsampled_input = self.upsampling_layer.forward(input)
        output = self.sequential_conv_blocks.forward(upsampled_input)
        if self.config.residual:
            output += upsampled_input
        return output

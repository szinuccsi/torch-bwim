import torch
import torch.nn as nn

from torch_bwim.nets.ActivationFunctions import ActivationFunctions


class ConvBlock1D(nn.Module):

    '''
        mode: 'same' or 'valid' with same meaning as mentioned in numpy documentation
        (https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
    '''
    def __init__(self,
                 in_ch, out_ch, kernel_size, activation_function, batch_norm=False,
                 stride=1, dilation=1, dropout_p=0.0, bias=True, mode='same',
                 padding_mode='replicate'):
        super().__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
            padding=self.get_padding_size(kernel_size=kernel_size, dilation=dilation, mode=mode),
            stride=stride, dilation=dilation,
            bias=bias and (not batch_norm), padding_mode=padding_mode
        )
        self.activation_function = activation_function if isinstance(activation_function, nn.Module) \
            else ActivationFunctions.get_function(activation_function)
        self._batch_norm = batch_norm
        self.batch_norm_layer = nn.BatchNorm1d(out_ch) if batch_norm else nn.Identity()
        self.dropout_layer = nn.Dropout(p=dropout_p)

    @classmethod
    def get_padding_size(cls, kernel_size, dilation, mode):
        if mode == 'valid':
            return 0
        elif mode == 'same':
            if kernel_size % 2 == 0:
                raise RuntimeWarning(f'kernel_size ({kernel_size}) is not even')
            return round(kernel_size // 2) * dilation
        else:
            raise RuntimeError(f'Invalid mode ({mode})')

    @property
    def batch_norm(self): return self._batch_norm

    def forward(self, x):
        y = self.dropout_layer(self.batch_norm_layer(self.activation_function(self.conv_layer(x))))
        return y

    def __call__(self, x):
        return self.forward(x)

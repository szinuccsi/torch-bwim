import torch
import torch.nn as nn

from torch_bwim.nets.NetBase import NetBase
from torch_bwim.nets.modules.cnn.ConvBlock1D import ConvBlock1D


class MultiConvLevel1D(NetBase):
    class Config(NetBase.Config):
        def __init__(self, in_ch, o_ch, start_feat, kernel_size, num_of_levels, layers_per_level, n_lin_type, dropout_p,
                     b_norm=False):
            super().__init__()
            self.in_ch = in_ch
            self.o_ch = o_ch
            self.start_feat = start_feat
            self.kernel_size = kernel_size
            self.num_of_levels = num_of_levels
            self.layers_per_level = layers_per_level
            self.n_lin_type = n_lin_type
            self.dropout_p = dropout_p
            self.b_norm = b_norm

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.config=config
        this = self.config
        in_ch = this.in_ch
        oCh=round(this.start_feat)

        self.featLayer = ConvBlock1D(in_ch=in_ch, out_ch=oCh, kernel_size=this.kernel_size,
                                        activation_function=this.n_lin_type, dropout_p=this.dropout_p)
        in_ch = oCh
        self.upChannelLayers = nn.ModuleList()
        self.convLayersInLevel = nn.ModuleList()
        for level in range(this.num_of_levels):
            oCh = in_ch * 2
            self.upChannelLayers.append(ConvBlock1D(in_ch=in_ch, out_ch=oCh, kernel_size=this.kernel_size,
                                        activation_function=this.n_lin_type, dropout_p=this.dropout_p))
            in_ch = oCh
            layers = []
            for l_in_level in range(1, this.layers_per_level):
                layers.append(ConvBlock1D(in_ch=in_ch, out_ch=oCh, kernel_size=this.kernel_size,
                                        activation_function=this.n_lin_type, dropout_p=this.dropout_p))
            self.convLayersInLevel.append(nn.Sequential(*layers))
        oCh = this.o_ch
        self.downChannelLayer = ConvBlock1D(in_ch=in_ch, out_ch=oCh, kernel_size=this.kernel_size,
                                        activation_function=this.n_lin_type, dropout_p=this.dropout_p)

    def forward(self, input):
        this = self.config
        input = self.featLayer(input)
        for i in range(this.num_of_levels):
            input = self.upChannelLayers[i](input)
            residual_plus = input
            input = self.convLayersInLevel[i](input)
            input += residual_plus
        output = self.downChannelLayer(input)
        return output

    def __call__(self, input):
        return self.forward(input)




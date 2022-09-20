import torch
import torch.nn as nn

from torch_bwim.nets.NetBase import NetBase
from torch_bwim.nets.modules.cnn.ConvBlock1D import ConvBlock1D


class MultiConvLevel1D(NetBase):
    class Config(NetBase.Config):
        def __init__(self, inCh, oCh, startFeat, kernelSize, numOfLevels, layersPerLevel, nLinType, dropout_p,
                     bNorm=False):
            super().__init__()
            self.inCh = inCh
            self.oCh = oCh
            self.startFeat = startFeat
            self.kernelSize = kernelSize
            self.numOfLevels = numOfLevels
            self.layersPerLevel = layersPerLevel
            self.nLinType = nLinType
            self.dropout_p = dropout_p
            self.bNorm = bNorm

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.config=config
        this = self.config
        inCh = this.inCh
        oCh=round(this.startFeat)

        self.featLayer = ConvBlock1D(in_ch=inCh, out_ch=oCh, kernel_size=this.kernelSize,
                                        activation_function=this.nLinType, dropout_p=this.dropout_p)
        inCh = oCh
        self.upChannelLayers = nn.ModuleList()
        self.convLayersInLevel = nn.ModuleList()
        for level in range(this.numOfLevels):
            oCh = inCh * 2
            self.upChannelLayers.append(ConvBlock1D(in_ch=inCh, out_ch=oCh, kernel_size=this.kernelSize,
                                        activation_function=this.nLinType, dropout_p=this.dropout_p))
            inCh = oCh
            layers = []
            for l_in_level in range(1, this.layersPerLevel):
                layers.append(ConvBlock1D(in_ch=inCh, out_ch=oCh, kernel_size=this.kernelSize,
                                        activation_function=this.nLinType, dropout_p=this.dropout_p))
            self.convLayersInLevel.append(nn.Sequential(*layers))
        oCh = this.oCh
        self.downChannelLayer = ConvBlock1D(in_ch=inCh, out_ch=oCh, kernel_size=this.kernelSize,
                                        activation_function=this.nLinType, dropout_p=this.dropout_p)

    def forward(self, input):
        this = self.config
        input = self.featLayer(input)
        for i in range(this.numOfLevels):
            input = self.upChannelLayers[i](input)
            residual_plus = input
            input = self.convLayersInLevel[i](input)
            input += residual_plus
        output = self.downChannelLayer(input)
        return output

    def __call__(self, input):
        return self.forward(input)




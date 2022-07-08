from torch_bwim.NetBase import NetBase


class MultiLayerNet(NetBase):

    class Config(object):
        def __init__(self, in_size: int, out_size: int, hidden_neurons: list, nonlinearity_type: str,
                     dropout_p=0.0, batch_norm=False):
            super().__init__()
            self.in_size = in_size
            self.out_size = out_size
            self.hidden_neurons = hidden_neurons
            self.nonlinearity_type = nonlinearity_type
            self.dropout_p = dropout_p
            self.batch_norm = batch_norm

    """linear->activation->dropout"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        this = config
        self.bias = not this.bNorm
        self.nonlinearity = NonLinearityType.create_nonlinearity(this.nLinType)
        self.linearLayers = nn.ModuleList()
        self.batchNormLayers = nn.ModuleList()
        for i in range(1, len(this.sizes)):
            self.linearLayers.append(nn.Linear(this.sizes[i - 1], this.sizes[i], bias=self.bias))
            if this.bNorm:
                self.batchNormLayers.append(nn.BatchNorm1d(this.sizes[i]))
            else:
                self.batchNormLayers.append(nn.Identity())
        if len(self.linearLayers) != len(self.batchNormLayers):
            raise RuntimeError('linearLayers - batchNormLayers not same number of layers: {lin} - {bNorm}'.format(
                lin=len(self.linearLayers), bNorm=len(self.batchNormLayers)
            ))
        self.dropoutLayer = nn.Dropout(p=this.dropout_p)

    def forward(self, x):
        y = x
        for i in range(len(self.linearLayers)):
            linearLayer = self.linearLayers[i]
            bNormLayer = self.batchNormLayers[i]
            y = linearLayer(y)
            y = self.nonlinearity(y)
            y = bNormLayer(y)
            y = self.dropoutLayer(y)
        return y

    def __call__(self, x):
        return self.forward(x)

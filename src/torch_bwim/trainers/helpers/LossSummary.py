import torch


class LossSummary(object):

    def __init__(self):
        super().__init__()
        self.sum_loss = 0.0
        self.counter = 0

    def add(self, loss, t: torch.Tensor=None, batch_size=None):
        if batch_size is None:
            batch_size = t.size(dim=0)
        self.sum_loss += batch_size * loss
        self.counter += batch_size

    def _get_loss(self): return self.sum_loss / self.counter
    loss = property(_get_loss)

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_bwim.interpolators.Interpolator1DFunction import Interpolator1DFunction
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class Interpolator1D(nn.Module):

    def __init__(self, xp: torch.Tensor, fp: torch.Tensor,
                 grad_fp: Optional[torch.Tensor] = None):
        super().__init__()
        self.xp = nn.Parameter(xp)
        self.fp = nn.Parameter(fp)

        self.grad_fp = torch.gradient(fp, spacing=(xp,))[0] if grad_fp is None else grad_fp

    def forward(self, x, left: Optional[torch.Tensor] = None, right: Optional[torch.Tensor] = None):
        original_shape = x.shape
        x = torch.flatten(x, start_dim=0)
        y = Interpolator1DFunction.apply(
            x, self.xp, self.fp,
            left, right, self.grad_fp
        )
        y = torch.reshape(y, original_shape)
        return y

    @classmethod
    def from_numpy(cls, xp: np.ndarray, fp: np.ndarray, fp_deriv: Optional[np.ndarray]=None):
        return Interpolator1D(
            xp=NnModuleUtils.from_array(xp), fp=NnModuleUtils.from_array(fp),
            grad_fp=NnModuleUtils.from_array(fp_deriv) if fp_deriv is not None else None
        )

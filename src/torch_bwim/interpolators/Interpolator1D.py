from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_bwim.interpolators.Interpolator1DFunction import Interpolator1DFunction
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class Interpolator1D(nn.Module):

    def __init__(self, xp: torch.Tensor, fp: torch.Tensor,
                 grad_fp: Optional[torch.Tensor]):
        super().__init__()
        self._xp = nn.Parameter(xp)
        self._fp = nn.Parameter(fp)
        self._left = torch.select(fp, dim=0, index=0).item()
        self._right = torch.select(fp, dim=0, index=-1).item()
        self._grad_fp = nn.Parameter(Interpolator1DFunction.gradient_create(xp=xp, fp=fp)
                                     if grad_fp is None else grad_fp)

    def forward(self, x, left: Optional[float] = None, right: Optional[float] = None):
        left = self._left if left is None else left
        right = self._right if right is None else right
        x = torch.flatten(x, start_dim=0)
        y = Interpolator1DFunction.apply(
            x, self._xp, self._fp,
            left, right, self._grad_fp
        )
        return y

    def __call__(self, x, left: Optional[torch.Tensor] = None, right: Optional[torch.Tensor] = None):
        return self.forward(x, left, right)

    @classmethod
    def from_numpy(cls, xp: np.ndarray, fp: np.ndarray, fp_deriv: Optional[np.ndarray]=None):
        return Interpolator1D(
            xp=NnModuleUtils.from_array(xp), fp=NnModuleUtils.from_array(fp),
            grad_fp=NnModuleUtils.from_array(fp_deriv) if fp_deriv is not None else None
        )

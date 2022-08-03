from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_bwim.interpolators.Interpolator1DFunction import Interpolator1DFunction
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class Interpolator1D(nn.Module):

    '''
        xp: shape(num of control points)
        fp: shape(num of control points, Optional[function cnt])
        grad_fp: shape(num of control points, Optional[function cnt])
    '''
    def __init__(self, xp: torch.Tensor, fp: torch.Tensor,
                 grad_fp: Optional[torch.Tensor]):
        super().__init__()
        fp, grad_fp = self.preprocess(fp=fp, xp=xp, grad_fp=grad_fp)
        self._xp = nn.Parameter(xp)
        self._fp = nn.Parameter(fp)
        self._grad_fp = nn.Parameter(grad_fp)
        self._left: Optional[nn.Parameter] = None
        self._right: Optional[nn.Parameter] = None
        self.postprocess(fp=fp)

    def preprocess(self, fp: torch.Tensor, xp: torch.Tensor,
                                  grad_fp: Optional[torch.Tensor]):
        with torch.no_grad():
            fp = fp.unsqueeze(-1) if len(fp.shape) == 1 else fp
            grad_fp = Interpolator1DFunction.gradient_create(xp=xp, fp=fp) \
                if grad_fp is None else grad_fp
            grad_fp = grad_fp.unsqueeze(-1) if len(grad_fp.shape) == 1 else grad_fp
        return fp, grad_fp

    def postprocess(self, fp: torch.Tensor):
        with torch.no_grad():
            self._left = nn.Parameter(torch.select(fp, dim=0, index=0))
            self._right = nn.Parameter(torch.select(fp, dim=0, index=-1))

    def forward(self, x, left: Optional[torch.Tensor] = None, right: Optional[torch.Tensor] = None):
        left = self._left if left is None else left
        right = self._right if right is None else right
        y = Interpolator1DFunction.apply(
            x, self._xp, self._fp,
            left, right, self._grad_fp
        )
        return y

    def append(self, fp: torch.Tensor, grad_fp: Optional[torch.Tensor]=None):
        fp, grad_fp = self.preprocess(fp=fp, xp=self._xp, grad_fp=grad_fp)
        self._fp = nn.Parameter(torch.cat([self._fp, fp], dim=-1))
        self._grad_fp = nn.Parameter(torch.cat([self._grad_fp, grad_fp], dim=-1))
        self.postprocess(fp=self._fp)

    def __call__(self, x, left: Optional[torch.Tensor] = None, right: Optional[torch.Tensor] = None):
        return self.forward(x, left, right)

    @classmethod
    def from_numpy(cls, xp: np.ndarray, fp: np.ndarray, fp_deriv: Optional[np.ndarray]=None):
        return Interpolator1D(
            xp=NnModuleUtils.from_array(xp), fp=NnModuleUtils.from_array(fp),
            grad_fp=NnModuleUtils.from_array(fp_deriv) if fp_deriv is not None else None
        )

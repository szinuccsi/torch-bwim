from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_bwim.interpolators.RectiBilinearInterpolateFunction import RectiBilinearInterpolateFunction
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class RectiBilinearInterpolate(nn.Module):

    def __init__(self, fp: torch.Tensor,
                 distinct_xp: torch.Tensor, distinct_yp: torch.Tensor,
                 grad_x_fp: Optional[torch.Tensor]=None, grad_y_fp: Optional[torch.Tensor]=None,
                 method='linear', fill_mode='fill'):
        super().__init__()
        self._distinct_xp = nn.Parameter(distinct_xp)
        self._distinct_yp = nn.Parameter(distinct_yp)

        fp, grad_x_fp, grad_y_fp = self.preprocess(
            fp=fp, grad_x_fp=grad_x_fp, grad_y_fp=grad_y_fp
        )
        self._fp = nn.Parameter(fp)
        self._grad_x_fp = nn.Parameter(grad_x_fp)
        self._grad_y_fp = nn.Parameter(grad_y_fp)

        self._method = method
        self._fill_mode = fill_mode

    def preprocess(self, fp: torch.Tensor,
                   grad_x_fp: Optional[torch.Tensor]=None,
                   grad_y_fp: Optional[torch.Tensor]=None):
        fp = fp.unsqueeze(-1) if len(fp.shape) == 2 else fp
        if (grad_x_fp is None) or (grad_y_fp is None):
            grad_x_fp, grad_y_fp = self.gradient_create(
                fp=fp, distinct_xp=self._distinct_xp, distinct_yp=self._distinct_yp
            )
        grad_x_fp = grad_x_fp.unsqueeze(-1) if len(grad_x_fp.shape) == 2 else grad_x_fp
        grad_y_fp = grad_y_fp.unsqueeze(-1) if len(grad_y_fp.shape) == 2 else grad_y_fp
        return fp, grad_x_fp, grad_y_fp

    @classmethod
    def gradient_create(cls, fp: torch.Tensor, distinct_xp: torch.Tensor, distinct_yp: torch.Tensor):
        return RectiBilinearInterpolateFunction.gradient_create(
            fp=fp, distinct_xp=distinct_xp, distinct_yp=distinct_yp
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x, y = torch.flatten(x, start_dim=0), torch.flatten(y, start_dim=0)
        return RectiBilinearInterpolateFunction.apply(
            x, y,
            self._fp, self._distinct_xp, self._distinct_yp,
            self._grad_x_fp, self._grad_y_fp, self._method,
            self._fill_mode
        )

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return self.forward(x, y)

    def _get_grid_cnt(self):return self._fp.shape[0]
    grid_cnt = property(_get_grid_cnt)

    def _get_distinct_xp(self): return self._distinct_xp
    distinct_xp = property(_get_distinct_xp)
    def _get_distinct_yp(self): return self._distinct_yp
    distinct_yp = property(_get_distinct_yp)

    def _get_method(self): return self._method
    def _set_method(self, val): self._method = val
    method = property(_get_method, _set_method)

    def append(self, fp: torch.Tensor,
               grad_x_fp: Optional[torch.Tensor]=None, grad_y_fp: Optional[torch.Tensor]=None):
        fp, grad_x_fp, grad_y_fp = self.preprocess(
            fp=fp, grad_x_fp=grad_x_fp, grad_y_fp=grad_y_fp
        )
        self._fp = nn.Parameter(torch.cat([self._fp, fp], dim=-1))
        self._grad_x_fp = nn.Parameter(torch.cat([self._grad_x_fp, grad_x_fp], dim=-1))
        self._grad_y_fp = nn.Parameter(torch.cat([self._grad_y_fp, grad_y_fp], dim=-1))
        return self

    @classmethod
    def merge(cls, interpolators: list):
        if len(interpolators) == 0:
            raise RuntimeError(f'len(interpolators)({len(interpolators)}) == 0')
        one_interpolator: RectiBilinearInterpolate = interpolators[0]
        distinct_xp, distinct_yp = one_interpolator.distinct_xp, one_interpolator.distinct_yp
        fp, grad_x_fp, grad_y_fp = [], [], []
        method = one_interpolator._method
        for i in range(len(interpolators)):
            interp: RectiBilinearInterpolate = interpolators[i]
            fp.append(interp._fp)
            grad_x_fp.append(interp._grad_x_fp)
            grad_y_fp.append(interp._grad_y_fp)

        fp = torch.cat(fp, dim=1)
        grad_x_fp = torch.cat(grad_x_fp, dim=1)
        grad_y_fp = torch.cat(grad_y_fp, dim=1)
        return RectiBilinearInterpolate(
            fp=fp,
            distinct_xp=distinct_xp, distinct_yp=distinct_yp,
            grad_x_fp=grad_x_fp, grad_y_fp=grad_y_fp,
            method=method
        )

    @classmethod
    def from_numpy(cls, fp: np.ndarray,
                   distinct_xp: np.ndarray, distinct_yp: np.ndarray,
                   grad_x_fp: Optional[np.ndarray]=None, grad_y_fp: Optional[np.ndarray]=None,
                   method='linear'):
        return RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(fp),
            distinct_xp=NnModuleUtils.from_array(distinct_xp),
            distinct_yp=NnModuleUtils.from_array(distinct_yp),
            grad_x_fp=NnModuleUtils.from_array(grad_x_fp) if grad_x_fp is not None else None,
            grad_y_fp=NnModuleUtils.from_array(grad_y_fp) if grad_y_fp is not None else None,
            method=method
        )


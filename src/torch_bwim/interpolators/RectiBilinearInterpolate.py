import torch
import torch.nn as nn

from torch_bwim.interpolators.RectiBilinearInterpolateFunction import RectiBilinearInterpolateFunction


class RectiBilinearInterpolate(nn.Module):

    def __init__(self, ctrl_values, distinct_xs, distinct_ys,
                ctrl_gradient_x=None, ctrl_gradient_y=None,
                method='linear'):
        super().__init__()
        self._ctrl_values = nn.Parameter(ctrl_values)
        self._distinct_xs = nn.Parameter(distinct_xs)
        self._distinct_ys = nn.Parameter(distinct_ys)
        if ctrl_gradient_x is None or ctrl_gradient_y is None:
            ctrl_gradient_x, ctrl_gradient_y = RectiBilinearInterpolateFunction.gradient_create(
                ctrl_values, distinct_xs, distinct_ys
            )
        self._ctrl_gradient_x = nn.Parameter(ctrl_gradient_x)
        self._ctrl_gradient_y = nn.Parameter(ctrl_gradient_y)
        self._method = method

    def forward(self, xs, ys):
        _forward = RectiBilinearInterpolateFunction.apply
        return _forward(xs, ys, self._ctrl_values,
                self._distinct_xs, self._distinct_ys,
                self._ctrl_gradient_x, self._ctrl_gradient_y,
                self._method)

    def __call__(self, xs, ys):
       return self.forward(xs, ys)

    def _get_surface_cnt(self):
        return self._ctrl_values.shape[0]

    surface_cnt = property(_get_surface_cnt)

    @classmethod
    def merge(cls, interpolators=[]):
        if len(interpolators) == 0:
            raise RuntimeError(f'len(interpolators)({len(interpolators)}) == 0')
        one_interpolator: RectiBilinearInterpolate = interpolators[0]
        distinct_xs, distinct_ys = one_interpolator._distinct_xs, one_interpolator._distinct_ys
        ctrl_values, ctrl_gradient_x, ctrl_gradient_y = [], [], []
        method = one_interpolator._method
        for i in range(len(interpolators)):
            interp: RectiBilinearInterpolate = interpolators[i]
            ctrl_values.append(interp._ctrl_values)
            ctrl_gradient_x.append(interp._ctrl_gradient_x)
            ctrl_gradient_y.append(interp._ctrl_gradient_y)
        ctrl_values = torch.cat(ctrl_values, dim=0)
        ctrl_gradient_x = torch.cat(ctrl_gradient_x, dim=0)
        ctrl_gradient_y = torch.cat(ctrl_gradient_y, dim=0)
        result = RectiBilinearInterpolate(
            ctrl_values=ctrl_values, distinct_xs=distinct_xs, distinct_ys=distinct_ys,
            ctrl_gradient_x=ctrl_gradient_x, ctrl_gradient_y=ctrl_gradient_y,
            method=method
        )
        return result

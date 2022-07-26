from typing import Optional

import numpy as np
import torch
from torch.autograd import Function


class RectiBilinearInterpolateFunction(Function):

    '''
        x: shape(num of points to interpolate)
        y: shape(num of points to interpolate)
        fp: shape(distinct y coords; distinct x coords; grid count)
        xp: (distinct y coords; distinct x coords)
        yp: (distinct y coords; distinct x coords)
        distinct_xp: (num of distinct x coords) - sorted
        distinct_yp: (num of distinct y coords) - sorted
        grad_x_fp: (grid count; distinct y coords; distinct x coords)
        grad_y_fp: (grid count; distinct y coords; distinct x coords)
        method: 'linear', 'farthest' or 'nearest'
    '''
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor,
                fp: torch.Tensor,
                distinct_xp: torch.Tensor, distinct_yp: torch.Tensor,
                grad_x_fp=None, grad_y_fp=None, method='linear'):
        ctx.save_for_backward(x, y)
        fp = torch.flatten(fp, start_dim=0, end_dim=-2)
        grid_cnt = fp.shape[-1]

        x_idxs_in_distinct = torch.searchsorted(distinct_xp, x, right=True)
        y_idxs_in_distinct = torch.searchsorted(distinct_yp, y, right=True)

        x_len, y_len = distinct_xp.shape[0], distinct_yp.shape[0]

        x_idxs_in_dictinct = [torch.clamp(x_idxs_in_distinct - 1.0, min=0, max=x_len - 1).int(),
                              torch.clamp(x_idxs_in_distinct, min=0, max=x_len - 1).int()]
        y_idxs_in_distinct = [torch.clamp(y_idxs_in_distinct - 1.0, min=0, max=y_len - 1).int(),
                              torch.clamp(y_idxs_in_distinct, min=0, max=y_len - 1).int()]

        xp_values = [torch.index_select(distinct_xp, dim=0, index=x_idxs_in_dictinct[i]) for i in range(2)]
        yp_values = [torch.index_select(distinct_yp, dim=0, index=y_idxs_in_distinct[i]) for i in range(2)]

        rectangle_idxs = [[x_idxs_in_dictinct[i] + y_idxs_in_distinct[j] * x_len for j in range(2)]
                          for i in range(2)]
        fp_values = [[torch.index_select(fp, dim=0, index=rectangle_idxs[i][j]) for j in range(2)]
                     for i in range(2)]

        w_interp = [[torch.abs((xp_values[1-i] - x) * (yp_values[1 - j] - y)) for j in range(2)]
                    for i in range(2)]
        w_interp = [[torch.stack([w_interp[i][j] for _ in range(grid_cnt)], dim=1) for j in range(2)]
                    for i in range(2)]
        ctx.distinct_xp, ctx.distinct_yp = distinct_xp, distinct_yp
        ctx.grad_x_fp, ctx.grad_y_fp = grad_x_fp, grad_y_fp
        ctx.fp = fp
        if method == 'linear':
            output = RectiBilinearInterpolateFunction.bilinear_interp(fp_values, w_interp)
            output = RectiBilinearInterpolateFunction.out_of_bounds_values(output, distinct_xp, x)
            output = RectiBilinearInterpolateFunction.out_of_bounds_values(output, distinct_yp, y)
            return output
        elif method == 'nearest':
            output = RectiBilinearInterpolateFunction.nearest_interp(fp_values, w_interp)
            return output
        elif method == 'farthest':
            output = RectiBilinearInterpolateFunction.farthest_interp(fp_values, w_interp)
            return output
        else:
            raise RuntimeError(f'Invalid interpolation method({method})')

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        fp = ctx.fp
        distinct_xp, distinct_yp = ctx.distinct_xp, ctx.distinct_yp
        grad_x_fp, grad_y_fp = ctx.grad_x_fp, ctx.grad_y_fp
        if (grad_x_fp is None) or (grad_y_fp is None):
            grad_x_fp, grad_y_fp = RectiBilinearInterpolateFunction.gradient_create(
                fp=fp, distinct_xp=distinct_xp, distinct_yp=distinct_yp
            )
        forward = RectiBilinearInterpolateFunction.apply
        gradient_x = forward(
            x, y, grad_x_fp, distinct_xp, distinct_yp,
            None, None, 'linear'
        )
        gradient_y = forward(
            x, y, grad_y_fp, distinct_xp, distinct_yp,
            None, None, 'linear'
        )
        gradient_x, gradient_y = torch.sum(gradient_x, dim=-1), torch.sum(gradient_y, dim=-1)
        return grad_output * gradient_x, grad_output * gradient_y, None, None, None, None, None, None

    @staticmethod
    def bilinear_interp(f, w):
        sum_w = (w[0][0] + w[0][1] + w[1][0] + w[1][1])
        res = (
            f[0][0] * w[0][0] +
            f[0][1] * w[0][1] +
            f[1][0] * w[1][0] +
            f[1][1] * w[1][1]
        ) / sum_w
        nans = torch.isnan(res)
        res = torch.where(nans, f[0][0], res)
        return res

    @staticmethod
    def nearest_interp(f, w):
        chosen_w = w[0][0]
        res = f[0][0]
        for i in range(2):
            for j in range(2):
                res = torch.where(chosen_w < w[i][j], f[i][j], res)
                chosen_w = torch.where(chosen_w < w[i][j], w[i][j], chosen_w)
        return res

    @staticmethod
    def farthest_interp(f, w):
        chosen_w = w[0][0]
        res = f[0][0]
        for i in range(2):
            for j in range(2):
                res = torch.where(w[i][j] < chosen_w, f[i][j], res)
                chosen_w = torch.where(w[i][j] < chosen_w, w[i][j], chosen_w)
        return res

    @staticmethod
    def out_of_bounds_values(t, distinct_coords, coords_to_interp):
        grid_cnt = t.shape[-1]
        distinct_coords = torch.stack([distinct_coords for _ in range(grid_cnt)], dim=1)
        coords_to_interp = torch.stack([coords_to_interp for _ in range(grid_cnt)], dim=1)

        t = torch.where(coords_to_interp < torch.min(distinct_coords), torch.zeros_like(t), t)
        t = torch.where(torch.max(distinct_coords) < coords_to_interp, torch.zeros_like(t), t)
        return t

    @staticmethod
    def gradient_create(fp: torch.Tensor, distinct_xp: torch.Tensor, distinct_yp: torch.Tensor):
        grad_x_fp, grad_y_fp = [], []
        for i in range(fp.shape[-1]):
            fp_for_one_surface = torch.select(fp, dim=-1, index=i)
            new_gradient_y, new_gradient_x = torch.gradient(fp_for_one_surface,
                                                            spacing=(distinct_yp, distinct_xp))
            new_gradient_x = new_gradient_x.unsqueeze(-1)
            new_gradient_y = new_gradient_y.unsqueeze(-1)
            grad_x_fp.append(new_gradient_x)
            grad_y_fp.append(new_gradient_y)
        grad_x_fp = torch.cat(grad_x_fp, dim=-1)
        grad_y_fp = torch.cat(grad_y_fp, dim=-1)
        return grad_x_fp, grad_y_fp

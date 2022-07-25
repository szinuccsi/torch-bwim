import numpy as np
import torch
from torch.autograd import Function


class RectiBilinearInterpolateFunction(Function):

    '''
        xs: shape: (num of points to interpolate)
        ys: shape: (num of points to interpolate)
        ctrl_xs: (distinct y coords, distinct x coords)
        ctrl_ys: (distinct y coords, distinct x coords)
        ctrl_values: (influence surfaces, distinct y coords, distinct x coords)
        distinct_xs: (num of distinct x coords) - sorted
        distinct_ys: (num of distinct y coords) - sorted
        ctrl_gradient_x: (influence surfaces, distinct y coords, distinct x coords)
        ctrl_gradient_y: (influence surfaces, distinct y coords, distinct x coords)
        method: 'linear' or 'nearest'
    '''
    @staticmethod
    def forward(ctx, xs, ys, ctrl_values,
                distinct_xs, distinct_ys,
                ctrl_gradient_x=None, ctrl_gradient_y=None,
                method='linear'):
        ctx.save_for_backward(xs, ys)
        ctx.ctrl_values = ctrl_values
        ctx.distinct_xs, ctx.distinct_ys = distinct_xs, distinct_ys
        ctx.ctrl_gradient_x, ctx.ctrl_gradient_y = ctrl_gradient_x, ctrl_gradient_y

        xs, ys = torch.flatten(xs, start_dim=0), torch.flatten(ys, start_dim=0)
        ctrl_values = torch.flatten(ctrl_values, start_dim=1)
        infl_surf_cnt = ctrl_values.shape[0]
        x_indices = RectiBilinearInterpolateFunction._searchsorted(distinct_xs, xs)
        y_indices = RectiBilinearInterpolateFunction._searchsorted(distinct_ys, ys)

        x_len = distinct_xs.shape[0]

        x_idxs = [
            torch.clamp(x_indices-1.0, min=0, max=len(distinct_xs)-1).int(),
            torch.clamp(x_indices, min=0, max=len(distinct_xs)-1).int()
        ]

        y_idxs = [
            torch.clamp(y_indices-1.0, min=0, max=len(distinct_ys)-1).int(),
            torch.clamp(y_indices, min=0, max=len(distinct_ys)-1).int()
        ]

        x = [torch.index_select(distinct_xs, dim=0, index=x_idxs[i]) for i in range(2)]
        y = [torch.index_select(distinct_ys, dim=0, index=y_idxs[i]) for i in range(2)]

        indices = [[x_idxs[i] + y_idxs[j] * x_len for j in range(2)] for i in range(2)]

        f = [[torch.index_select(ctrl_values, dim=1, index=indices[i][j])
                     for j in range(2)] for i in range(2)]

        w_interp = [[torch.abs((x[1-i] - xs) * (y[1-j] - ys)) for j in range(2)] for i in range(2)]
        w_interp = [[torch.stack([w_interp[i][j] for _ in range(infl_surf_cnt)]) for j in range(2)] for i in range(2)]
        ctx.w_interp = w_interp
        if method == 'linear':
            output = RectiBilinearInterpolateFunction._bilinear_interp(f, w_interp)
            output = RectiBilinearInterpolateFunction._out_of_bounds_values(output, distinct_xs, xs)
            output = RectiBilinearInterpolateFunction._out_of_bounds_values(output, distinct_ys, ys)
            return output
        elif method == 'nearest':
            output = RectiBilinearInterpolateFunction._nearest_interp(f, w_interp)
            return output
        elif method == 'farthest':
            output = RectiBilinearInterpolateFunction._farthest_interp(f, w_interp)
            return output
        else:
            raise RuntimeError(f'Invalid interpolation method({method})')

    @staticmethod
    def backward(ctx, grad_output):
        w_interp = ctx.w_interp
        xs, ys = ctx.saved_tensors
        ctrl_values = ctx.ctrl_values
        distinct_xs, distinct_ys = ctx.distinct_xs, ctx.distinct_ys
        ctrl_gradient_x, ctrl_gradient_y = ctx.ctrl_gradient_x, ctx.ctrl_gradient_y
        if (ctrl_gradient_x is None) or (ctrl_gradient_y is None):
            ctrl_gradient_x, ctrl_gradient_y = RectiBilinearInterpolateFunction.gradient_create(
                ctrl_values, distinct_xs, distinct_ys
            )
        RectiBilinearInterpolateFunction._check_backward(distinct_xs, distinct_ys, ctrl_values,
                                                         ctrl_gradient_x, ctrl_gradient_y)
        forward = RectiBilinearInterpolateFunction.apply
        gradient_x = forward(xs, ys, ctrl_gradient_x,
                             distinct_xs, distinct_ys,
                             None, None, 'linear')
        gradient_y = forward(xs, ys, ctrl_gradient_y,
                             distinct_xs, distinct_ys,
                             None, None, 'linear')
        return grad_output * gradient_x, grad_output * gradient_y, None, None, None, None, None, None

    @staticmethod
    def gradient_create(ctrl_values, distinct_xs, distinct_ys):
        ctrl_gradient_x, ctrl_gradient_y = [], []
        ctrl_values = ctrl_values.numpy()
        distinct_ys = distinct_ys.numpy()
        distinct_xs = distinct_xs.numpy()
        for i in range(ctrl_values.shape[0]):
            ctrl_values_for_one_surface = ctrl_values[i]
            new_gradient_y, new_gradient_x = np.gradient(ctrl_values_for_one_surface, distinct_ys, distinct_xs)
            ctrl_gradient_x.append(new_gradient_x)
            ctrl_gradient_y.append(new_gradient_y)
        ctrl_gradient_x = torch.from_numpy(np.asarray(ctrl_gradient_x).astype(np.float32))
        ctrl_gradient_y = torch.from_numpy(np.asarray(ctrl_gradient_y).astype(np.float32))
        return ctrl_gradient_x, ctrl_gradient_y

    @staticmethod
    def _check_forward(xs, ys, ctrl_values,
                       distinct_xs, distinct_ys):
        if (len(xs.shape) != 1) or (len(ys.shape) != 1):
            raise RuntimeError(f'(len(xs.shape)({len(xs.shape)}) != 1) or'
                               f' (len(ys.shape)({len(ys.shape)}) != 1)')
        if xs.shape[0] != ys.shape[0]:
            raise RuntimeError(f'xs.shape[0]({xs.shape[0]}) != ys.shape[0]({ys.shape[0]})')
        if (len(distinct_xs.shape) != 1) or (len(distinct_ys.shape) != 1):
            raise RuntimeError(f'(len(distinct_xs.shape)({len(distinct_xs.shape)}) != 1) or '
                               f'(len(distinct_ys.shape)({len(distinct_ys.shape)}) != 1)')
        num_of_distinct_xs, num_of_distinct_ys = distinct_xs.shape[0], distinct_ys.shape[0]
        if (len(ctrl_values.shape) != 3):
            raise RuntimeError(f'(len(ctrl_values.shape)({len(ctrl_values.shape)}) != 3)')
        if (ctrl_values.shape[1] != num_of_distinct_ys) or (ctrl_values.shape[2] != num_of_distinct_xs):
            raise RuntimeError(f'ctrl_values.shape({ctrl_values.shape}) != '
                               f'(_, {num_of_distinct_ys}, {num_of_distinct_xs})')

    @staticmethod
    def _check_backward(distinct_xs, distinct_ys, ctrl_values, ctrl_gradient_x, ctrl_gradient_y):
        num_of_distinct_xs = distinct_xs.shape[0]
        num_of_distinct_ys = distinct_ys.shape[0]
        infl_surf_cnt = ctrl_values.shape[0]
        if  (len(ctrl_gradient_x.shape) != 3) or (len(ctrl_gradient_y.shape) != 3):
            raise RuntimeError(f'(len(ctrl_gradient_x)({len(ctrl_gradient_x.shape)}) != 3) or '
                               f'(len(ctrl_gradient_y)({len(ctrl_gradient_y.shape)}) != 3)')
        if (ctrl_gradient_x.shape[0] != infl_surf_cnt) or \
                (ctrl_gradient_x.shape[1] != num_of_distinct_ys) or \
                (ctrl_gradient_x.shape[2] != num_of_distinct_xs):
            raise RuntimeError(f'ctrl_gradient_x.shape({ctrl_gradient_x.shape}) != '
                               f'({infl_surf_cnt}, {num_of_distinct_ys}, {num_of_distinct_xs})')
        if (ctrl_gradient_y.shape[0] != infl_surf_cnt) or \
                (ctrl_gradient_y.shape[1] != num_of_distinct_ys) or \
                (ctrl_gradient_y.shape[2] != num_of_distinct_xs):
            raise RuntimeError(f'ctrl_gradient_y.shape({ctrl_gradient_y.shape}) != '
                               f'({infl_surf_cnt}, {num_of_distinct_ys}, {num_of_distinct_xs})')

    @staticmethod
    def _bilinear_interp(f, w):
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
    def _nearest_interp(f, w):
        chosen_w = w[0][0]
        res = f[0][0]
        for i in range(2):
            for j in range(2):
                res = torch.where(chosen_w < w[i][j], f[i][j], res)
                chosen_w = torch.where(chosen_w < w[i][j], w[i][j], chosen_w)
        return res

    @staticmethod
    def _farthest_interp(f, w):
        chosen_w = w[0][0]
        res = f[0][0]
        for i in range(2):
            for j in range(2):
                res = torch.where(w[i][j] < chosen_w, f[i][j], res)
                chosen_w = torch.where(w[i][j] < chosen_w, w[i][j], chosen_w)
        return res

    @staticmethod
    def _out_of_bounds_values(t, distinct_coords, coords_to_interp):
        t = torch.where(coords_to_interp < torch.min(distinct_coords), torch.zeros_like(t), t)
        t = torch.where(torch.max(distinct_coords) < coords_to_interp, torch.zeros_like(t), t)
        return t

    @staticmethod
    def _searchsorted(disctinct_coords, coords_to_find):
        return torch.searchsorted(disctinct_coords, coords_to_find, right=True)

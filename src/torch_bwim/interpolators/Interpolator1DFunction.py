import torch
from torch.autograd import Function


class Interpolator1DFunction(Function):

    '''
        kernel_xs: shape(kernel_size) sorted
        kernel: shape(batch, kernel_size)
        xs: shape(kernel_size)
        fill_value: 'edge' or constant
    '''
    @staticmethod
    def forward(ctx, xs, ctrl_xs, ctrl_values, method='linear', fill_value='edge'):
        ctx.save_for_backward(xs)

        indices = torch.searchsorted(ctrl_xs, xs, right=True)

        indices = [
            torch.clamp(indices - 1.0, min=0, max=len(ctrl_xs) - 1).int(),
            torch.clamp(indices, min=0, max=len(ctrl_xs) - 1).int()
        ]

        w_interp = [torch.abs(torch.index_select(ctrl_xs, dim=0, index=indices[1 - i]) - xs) for i in range(2)]
        w_interp = Interpolator1DFunction._normalize_w_interp(w_interp=w_interp)
        f_values = [torch.index_select(ctrl_values, dim=1, index=indices[i]) for i in range(2)]

        output = Interpolator1DFunction._interpolate(w_interp=w_interp, f_values=f_values)
        output = Interpolator1DFunction._out_of_bounds_interpolate(
            xs=xs, kernel_xs=ctrl_xs, values=output, fill_value=fill_value
        )
        ctx.indices = indices
        ctx.w_interp = w_interp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass

    @staticmethod
    def _normalize_w_interp(w_interp):
        sum_w = w_interp[0] + w_interp[1]
        w_interp = [w_interp[i] / sum_w for i in range(2)]
        nans = [torch.isnan(w_interp[i]) for i in range(2)]
        w_interp = [torch.where(nans[i], torch.full_like(w_interp[i], 0.5), w_interp[i])
                    for i in range(2)]
        return w_interp

    @staticmethod
    def _interpolate(w_interp, f_values):
        return w_interp[0] * f_values[0] + w_interp[1] * f_values[1]

    @staticmethod
    def _out_of_bounds_interpolate(xs, kernel_xs, values, fill_value):
        if fill_value == 'edge':
            return values
        min_x, max_x = torch.min(kernel_xs), torch.max(kernel_xs)
        values = torch.where(xs < min_x, torch.full_like(values, fill_value=fill_value), values)
        values = torch.where(max_x < xs, torch.full_like(values, fill_value=fill_value), values)
        return values

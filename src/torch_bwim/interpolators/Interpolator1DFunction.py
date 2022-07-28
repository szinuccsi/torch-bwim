from typing import Optional

import torch
from torch.autograd import Function


class Interpolator1DFunction(Function):

    '''
        x: shape(batch_size)
        xp: shape(num of control points)
        fp: shape(num of control points)
        grad_fp: shape(num of control points)

        return: shape(num of control points)
    '''
    @staticmethod
    def forward(ctx, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor,
                left: Optional[float], right: Optional[float],
                grad_fp: Optional[torch.Tensor]) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx._grad_fp = grad_fp
        indices = torch.searchsorted(xp, x, right=True)
        left = torch.select(fp, dim=0, index=0).item()
        right = torch.select(fp, dim=0, index=-1).item()
        indices = [
            torch.clamp(indices - 1.0, min=0, max=len(xp) - 1).int(),
            torch.clamp(indices, min=0, max=len(xp) - 1).int()
        ]

        w_interp = [torch.abs(torch.index_select(xp, dim=0, index=indices[1 - i]) - x) for i in range(2)]
        w_interp = Interpolator1DFunction.normalize_w_interp(w_interp=w_interp)
        fp_values = [torch.index_select(fp, dim=0, index=indices[i]) for i in range(2)]

        output = w_interp[0] * fp_values[0] + w_interp[1] * fp_values[1]
        output = Interpolator1DFunction.out_of_bounds_interpolate(
            x=x, f=output, xp=xp, left=left, right=right
        )
        ctx._grad_fp = grad_fp
        ctx._xp, ctx._fp = xp, fp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        xp, fp = ctx._xp, ctx._fp
        grad_fp = ctx._grad_fp
        if grad_fp is None:
            grad_fp = Interpolator1DFunction.gradient_create(xp=xp, fp=fp)
        output = Interpolator1DFunction.apply(x, xp, grad_fp, None, None, grad_fp)
        return grad_output * output, None, None, None, None, None

    @staticmethod
    def normalize_w_interp(w_interp):
        sum_w = w_interp[0] + w_interp[1]
        w_interp = [w_interp[i] / sum_w for i in range(2)]
        nans = [torch.isnan(w_interp[i]) for i in range(2)]
        w_interp = [torch.where(nans[i], torch.full_like(w_interp[i], 0.5), w_interp[i])
                    for i in range(2)]
        return w_interp

    @staticmethod
    def out_of_bounds_interpolate(x: torch.Tensor, f: torch.Tensor, xp: torch.Tensor,
                                  left: float, right: float):
        device = x.device
        min_x, max_x = torch.min(xp), torch.max(xp)
        f = torch.where(x < min_x, torch.full_like(f, fill_value=left, device=device), f)
        f = torch.where(max_x < x, torch.full_like(f, fill_value=right, device=device), f)
        return f

    '''
        xp: shape(num of control points)
        fp: shape(num of control points)
        
        return: shape(num of control points)
    '''
    @staticmethod
    def gradient_create(xp: torch.Tensor, fp: torch.Tensor):
        grad_fp = torch.gradient(fp, spacing=(xp,))[0]
        print(grad_fp.shape)
        return grad_fp

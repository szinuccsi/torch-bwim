from typing import Optional

import torch
from torch.autograd import Function


class Interpolator1DFunction(Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor,
                left: Optional[torch.Tensor], right: Optional[torch.Tensor],
                grad_fp: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.grad_fp = grad_fp

        indices = torch.searchsorted(xp, x, right=True)
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
        ctx.indices = indices
        ctx.w_interp = w_interp
        ctx.grad_fp = grad_fp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.indices
        w_interp = ctx.w_interp
        grad_fp = ctx.grad_fp

        grad_fp_values = [torch.index_select(grad_fp, dim=0, index=indices[i]) for i in range(2)]
        output = w_interp[0] * grad_fp_values[0] + w_interp[1] * grad_fp_values[1]
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
                                  left: Optional[torch.Tensor], right: Optional[torch.Tensor]):
        min_x, max_x = torch.min(xp), torch.max(xp)
        if left is not None:
            f = torch.where(x < min_x, torch.full_like(f, fill_value=left), f)
        if right is not None:
            f = torch.where(max_x < x, torch.full_like(f, fill_value=right), f)
        return f

from typing import Optional

import torch
from torch.autograd import Function


class Interpolator1DFunction(Function):

    '''
        x: shape(batch_size)
        xp: shape(num of control points)
        fp: shape(num of control points, function cnt)
        left: shape(function cnt)
        right: shape(function cnt)
        grad_fp: shape(num of control points, function cnt)
        return: shape(num of control points, function_cnt)
    '''
    @staticmethod
    def forward(ctx, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor,
                left: Optional[torch.Tensor], right: Optional[torch.Tensor],
                grad_fp: Optional[torch.Tensor]) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx._grad_fp = grad_fp
        function_cnt = fp.shape[-1]
        indices = torch.searchsorted(xp, x, right=True)
        left = torch.select(fp, dim=0, index=0) if left is None else left
        right = torch.select(fp, dim=0, index=-1) if right is None else right
        indices = [
            torch.clamp(indices - 1.0, min=0, max=len(xp) - 1).int(),
            torch.clamp(indices, min=0, max=len(xp) - 1).int()
        ]

        w_interp = [torch.abs(torch.index_select(xp, dim=0, index=indices[1 - i]) - x) for i in range(2)]
        w_interp = Interpolator1DFunction.normalize_w_interp(w_interp=w_interp)
        w_interp = [w_interp[i].unsqueeze(-1).expand(-1, function_cnt) for i in range(2)]

        fp_values = [torch.index_select(fp, dim=0, index=indices[i]) for i in range(2)]
        output = w_interp[0] * fp_values[0] + w_interp[1] * fp_values[1]
        output = Interpolator1DFunction.out_of_bounds_interpolate(
            x=x, f=output, xp=xp, left=left, right=right
        )
        ctx.grad_fp = grad_fp
        ctx.xp, ctx.fp = xp, fp
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        xp, fp = ctx.xp, ctx.fp
        grad_fp = ctx.grad_fp
        if grad_fp is None:
            grad_fp = Interpolator1DFunction.gradient_create(xp=xp, fp=fp)
        output = Interpolator1DFunction.apply(x, xp, grad_fp, None, None, None)
        return torch.sum(grad_output * output, dim=-1), None, None, None, None, None

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
                                  left: torch.Tensor, right: torch.Tensor):
        device = x.device
        x = x.unsqueeze(-1).expand(-1, f.shape[1])
        left = left.unsqueeze(0).expand(f.shape[0], f.shape[1])
        right = right.unsqueeze(0).expand(f.shape[0], f.shape[1])
        min_x, max_x = torch.min(xp), torch.max(xp)
        f = torch.where(x < min_x, left, f)
        f = torch.where(max_x < x, right, f)
        return f

    '''
        xp: shape(num of control points, function cnt)
        fp: shape(num of control points, function cnt)
        
        return: shape(num of control points)
    '''
    @staticmethod
    def gradient_create(xp: torch.Tensor, fp: torch.Tensor):
        grad_fp = []
        for i in range(fp.shape[-1]):
            fp_for_one_function = torch.select(fp, dim=-1, index=i)
            new_grad_fp = torch.gradient(fp_for_one_function, spacing=(xp,))[0]
            grad_fp.append(new_grad_fp)
        grad_fp = torch.stack(grad_fp, dim=-1)
        return grad_fp

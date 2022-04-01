import torch

from typing import Any


class DiffRoundFunction(torch.autograd.Function):
    """
    A differentiable round function.
    The next layer gradients are passed through backward.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        grad_input = grad_output.clone()
        return grad_input


class DiffRoundOrd3Function(torch.autograd.Function):
    """
    A differentiable round function.
    Approximate `round(x)` using `round(x) + (x - round(x)) ** 3`.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        input, = ctx.saved_tensors
        grad_input = 3 * ((input - torch.round(input)) ** 2)
        return grad_input * grad_output


diff_round = DiffRoundFunction.apply
diff_round_ord3 = DiffRoundOrd3Function.apply

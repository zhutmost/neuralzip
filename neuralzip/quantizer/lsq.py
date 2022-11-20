import math

import torch as t

from .quantizer import Quantizer
from .helper import *


def grad_scale(x: t.Tensor, scale: float) -> t.Tensor:
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x: t.Tensor) -> t.Tensor:
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LearnedStepQuantizer(Quantizer):
    """LSQ quantizer, based on LSQ-Net proposed by Steven K. Esser from IBM.

    Paper link: https://arxiv.org/abs/1902.08153.
    """

    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False) -> None:
        """Create an LSQ quantizer.

        Args:
            bit (int): Bit width of quantized weight.
            all_positive (bool): Whether to use symmetric quantization.
            symmetric (bool): Whether to quantize all the numbers to non-negative.
        """
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)

        self.alpha = t.nn.Parameter(t.tensor(1.))

        # a flag to indicate whether `alpha` is initialized
        self.register_buffer('initialized', t.zeros(1))

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training and self.initialized == 0:
            # initialize alpha with the first batch of input data
            self.alpha.data.copy_(2. * x.abs().mean() / math.sqrt(self.upper_bound))
            self.initialized.fill_(1)

        grad_scale_factor = math.sqrt(1. / (x.numel() * self.upper_bound))
        alpha = grad_scale(self.alpha, grad_scale_factor)

        x = (x / alpha).clamp(min=self.lower_bound, max=self.upper_bound)
        x = round_pass(x) * alpha
        return x

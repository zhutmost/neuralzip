from abc import ABC

import torch as t

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LSQQuantizer(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False):
        super().__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.s = t.nn.Parameter(t.tensor(1.))

        # a flag to indicate whether `s` is initialized
        self.register_buffer('initialized', t.tensor(0))

    def s_initialize(self, x):
        self.s = t.nn.Parameter(x.detach().clone().abs().mean() * 2. / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.training and not self.initialized.bool().item():
            self.s_initialize(x)
            self.initialized.fill_(1)

        s_grad_scale = 1. / ((self.thd_pos * x.numel()) ** 0.5)
        s_scaled = grad_scale(self.s, s_grad_scale)

        x = x / s_scaled
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scaled
        return x

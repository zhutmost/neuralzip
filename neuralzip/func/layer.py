from typing import Optional

import torch as t

from neuralzip.quantizer import Quantizer


class NzConv2d(t.nn.Conv2d):
    """Conv2d with injected quantizers."""

    def __init__(self,
                 m: t.nn.Conv2d,
                 quan_w_fn: Optional[Quantizer] = None,
                 quan_a_fn: Optional[Quantizer] = None) -> None:
        assert type(m) == t.nn.Conv2d
        super().__init__(in_channels=m.in_channels,
                         out_channels=m.out_channels,
                         kernel_size=m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: t.Tensor) -> t.Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight, self.bias)


class NzLinear(t.nn.Linear):
    """Linear with injected quantizers."""

    def __init__(self,
                 m: t.nn.Linear,
                 quan_w_fn: Optional[Quantizer] = None,
                 quan_a_fn: Optional[Quantizer] = None) -> None:
        assert type(m) == t.nn.Linear
        super().__init__(in_features=m.in_features,
                         out_features=m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: t.Tensor) -> t.Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)

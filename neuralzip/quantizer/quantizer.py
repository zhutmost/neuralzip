import torch as t


class Quantizer(t.nn.Module):
    """Base class of quantizers."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        raise NotImplementedError


class IdentityQuantizer(Quantizer):
    """Returns the input data as output without quantization.

    This quantizer is often used as a placeholder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x

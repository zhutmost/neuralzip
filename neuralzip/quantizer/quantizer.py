import torch as t


class Quantizer(t.nn.Module):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class IdentityQuantizer(Quantizer):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

import torch as t

from .layer import NzConv2d, NzLinear

DefaultQuantizedModuleMapping = {
    t.nn.Conv2d: NzConv2d,
    t.nn.Linear: NzLinear
}

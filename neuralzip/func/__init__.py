from typing import Type, Union

import torch as t

from .layer import NzConv2d, NzLinear

NZ_MODULE_T = Union[Type[NzConv2d], Type[NzLinear]]

DefaultQuantizedModuleMapping = {
    t.nn.Conv2d: NzConv2d,
    t.nn.Linear: NzLinear
}

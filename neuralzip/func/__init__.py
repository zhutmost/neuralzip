from typing import Callable, Dict, Type, Union

import torch as t

from .layer import NzConv2d, NzLinear

QUAN_MODULE_MAPPING_TYPE = Dict[Type[t.nn.Module], Type[t.nn.Module]]
DefaultQuantizedModuleMapping: QUAN_MODULE_MAPPING_TYPE = {
    t.nn.Conv2d: NzConv2d,
    t.nn.Linear: NzLinear
}

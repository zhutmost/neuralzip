from typing import Dict, Type, Tuple

from omegaconf import OmegaConf, DictConfig
import torch as t

from apputil import load_obj
from .func import DefaultQuantizedModuleMapping, QUAN_MODULE_MAPPING_TYPE
from .quantizer import Quantizer, IdentityQuantizer


def quantizer(cfg_quantizer: DictConfig) -> Quantizer:
    c = dict(cfg_quantizer)
    if c['class_name']:
        q = load_obj(c['class_name'], default_obj_path='neuralzip.quantizer')
    else:
        q = IdentityQuantizer
    return q(**c['params'])


def _replace_module_by_names(model: t.nn.Module,
                             modules_to_replace: Dict[str, t.nn.Module],
                             quantized_module_mapping: QUAN_MODULE_MAPPING_TYPE) -> t.nn.Module:
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in quantized_module_mapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model


def quantizer_inject(
        model: t.nn.Module,
        cfg_quan: DictConfig,
        quantized_module_mapping: QUAN_MODULE_MAPPING_TYPE = DefaultQuantizedModuleMapping
) -> t.nn.Module:
    # Find modules to quantize
    modules_to_replace = dict()
    for name, module in model.named_modules():
        if type(module) in quantized_module_mapping.keys():
            if name in cfg_quan.excepts:
                cfg_quan_weight = OmegaConf.merge(cfg_quan.weight, cfg_quan.excepts[name].weight)
                cfg_quan_act = OmegaConf.merge(cfg_quan.act, cfg_quan.excepts[name].act)
            else:
                cfg_quan_weight = cfg_quan.weight
                cfg_quan_act = cfg_quan.act
            if cfg_quan_weight['class_name'] or cfg_quan_act['class_name']:
                mapped_module = quantized_module_mapping[type(module)]
                modules_to_replace[name] = mapped_module(
                    module,
                    quan_w_fn=quantizer(cfg_quan_weight),
                    quan_a_fn=quantizer(cfg_quan_act)
                )
        elif name in cfg_quan.excepts:
            raise KeyError('Cannot find module %s in the model', name)

    quantized_model = _replace_module_by_names(model, modules_to_replace, quantized_module_mapping)
    return quantized_model


def quantizer_stat(model: t.nn.Module) -> Tuple[int, Dict[Type[Quantizer], int]]:
    quan_dict = dict()
    quan_cnt = 0
    for _, m in model.named_modules():
        if isinstance(m, Quantizer):
            quan_cnt += 1
            quan_dict[type(m)] = quan_dict.get(type(m), 0) + 1
    return quan_cnt, quan_dict

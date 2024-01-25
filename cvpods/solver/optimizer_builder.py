#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from typing import Any, Dict, List, Set

import torch
from torch import optim

from cvpods.utils.registry import Registry
from .lars_sgd import LARS_SGD

OPTIMIZER_BUILDER = Registry("Optimizer builder")

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)
from .optim_factory import LayerDecayValueAssigner

def exclude_from_wd(named_params, weight_decay, skip_list=['bias', 'bn']):
    params = []
    excluded_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {'params': params, 'weight_decay': weight_decay},
        {'params': excluded_params, 'weight_decay': 0., 'lars_exclude': True},
    ]


@OPTIMIZER_BUILDER.register()
class OptimizerBuilder:

    @staticmethod
    def build(model, cfg):
        raise NotImplementedError


@OPTIMIZER_BUILDER.register()
class SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM,
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class D2SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class LARS_SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        exclude = cfg.SOLVER.OPTIMIZER.get("WD_EXCLUDE_BN_BIAS", False)
        if exclude:
            param = exclude_from_wd(
                model.named_parameters(), cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
            )
        else:
            param = model.parameters()
        optimizer = LARS_SGD(
            param,
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            nesterov=cfg.SOLVER.OPTIMIZER.get("NESTERROV", False),
            eta=cfg.SOLVER.OPTIMIZER.TRUST_COEF,
            eps=cfg.SOLVER.OPTIMIZER.EPS,
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AdamBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


def get_fpn_model_parameters(
    model,
    weight_decay=1e-5,
    weight_decay_norm=0.0,
    base_lr=4e-5,
    skip_list=(),
    multiplier=1.5,
):
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        elif "norm" in name and weight_decay_norm is not None:
            group_name = "decay"
            this_weight_decay = weight_decay_norm
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if name.startswith("backbone.bottom_up.encoder.patch_embed"):
            group_name = "backbone.bottom_up.encoder.patch_embed_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr,
                }
        elif name.startswith("backbone.bottom_up.encoder"):
            group_name = "backbone.bottom_up.encoder_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr / multiplier,
                }
        else:
            group_name = "others_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr * multiplier,
                }

        parameter_group_vars[group_name]["params"].append(param)
    return list(parameter_group_vars.values())


def get_convnextv2_model_parameters(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

@OPTIMIZER_BUILDER.register()
class AdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR

        if 'get_fpn_model_parameters' in cfg.SOLVER.OPTIMIZER:
            assert isinstance(cfg.SOLVER.OPTIMIZER.get_fpn_model_parameters,dict)
            get_para_dict = cfg.SOLVER.OPTIMIZER.get_fpn_model_parameters
            params = get_fpn_model_parameters(
                model,
                weight_decay=get_para_dict['weight_decay'],
                weight_decay_norm=get_para_dict['weight_decay_norm'],
                base_lr=lr,
                skip_list=get_para_dict['skip_list'],
                multiplier=get_para_dict['multiplier'],
            )
            weight_decay = get_para_dict['weight_decay']
        elif 'get_convnextv2_model_parameters' in cfg.SOLVER.OPTIMIZER:
            get_para_dict = cfg.SOLVER.OPTIMIZER.get_convnextv2_model_parameters
            if get_para_dict['skip_list'] is not None:
                skip = get_para_dict['skip_list']
            elif hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            else:
                skip = None

            if 'layer_decay' in get_para_dict:
                layer_decay = get_para_dict['layer_decay']
                assert 'layer_decay_type' in get_para_dict
                layer_decay_type = get_para_dict['layer_decay_type'] 
                assert layer_decay_type in ['single', 'group']
                if layer_decay_type == 'group': # applies for Base and Large models
                    num_layers = 12
                else:
                    num_layers = sum(model.depths)
                assigner = LayerDecayValueAssigner(
                    list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
                    depths=model.depths, layer_decay_type=layer_decay_type)
            else:
                assigner = None

            params = get_convnextv2_model_parameters(model, 
                                          weight_decay=get_para_dict['weight_decay'],
                                          skip_list=skip,
                                          get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                          get_layer_scale=assigner.get_scale if assigner is not None else None)
            weight_decay = 0.
        else:
            params = model.parameters()
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=weight_decay,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class SGDGateLRBuilder(OptimizerBuilder):
    """
    SGD Gate LR optimizer builder, used for DynamicRouting in cvpods.
    This optimizer will ultiply lr for gating function.
    """

    @staticmethod
    def build(model, cfg):
        gate_lr_multi = cfg.SOLVER.OPTIMIZER.GATE_LR_MULTI
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

                if gate_lr_multi > 0.0 and "gate_conv" in name:
                    lr *= gate_lr_multi

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer

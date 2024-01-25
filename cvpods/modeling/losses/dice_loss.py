#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import torch
from torch import nn


def dice_loss(input, target):
    r"""
    Dice loss defined in the V-Net paper as:

    Loss_dice = 1 - D

            2 * sum(p_i * g_i)
    D = ------------------------------
         sum(p_i ^ 2) + sum(g_i ^ 2)

    where the sums run over the N mask pixels (i = 1 ... N), of the predicted binary segmentation
    pixel p_i ∈ P and the ground truth binary pixel g_i ∈ G.

    Args:
        input (Tensor): predicted binary mask, each pixel value should be in range [0, 1].
        target (Tensor): ground truth binary mask.

    Returns:
        Tensor: dice loss.
    """
    assert input.shape[-2:] == target.shape[-2:]
    input = input.view(input.size(0), -1).float()
    target = target.view(target.size(0), -1).float()

    d = (
        2 * torch.sum(input * target, dim=1)
    ) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4
    )

    return 1. - d.mean()




class DSCLoss(nn.Module):

    def __init__(self, alpha=0.0,reduce='mean'):
        super(DSCLoss, self).__init__()
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, input, target):
        assert input.shape[-2:] == target.shape[-2:]
        input = input.view(input.size(0), -1).float()
        target = target.view(target.size(0), -1).float()

        input_with_weight = input * ((1 - input) ** self.alpha)
        d = (2 * input_with_weight * target + 1e-4) / (
            input_with_weight * input + target + 1e-4
        )
        ## OR
        # d = (2 * input_with_weight * target + 1e-4) / (
        #     input_with_weight * input + target*target + 1e-4
        # )
        return (1 - d).mean()
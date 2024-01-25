#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from ..registry import PATH_ROUTES

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

_PREDEFINED_SPLITS_DIAOWANG = {
    "dataset_type": "DiaoWangDataset",
    "diaowang": {
        "DIAOWANG_2022_new_Cross_Individual_train":
            ("datasets/diaowang/", "datasets/new_Cross_Individual_train.json"),
        "DIAOWANG_final_cross_anns":
        ("datasets/diaowang/", "datasets/final_cross_anns.json"),

    },
    
}

PATH_ROUTES.register(_PREDEFINED_SPLITS_DIAOWANG,'DIAOWANG')
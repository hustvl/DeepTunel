#!/bin/bash
uname -a
#date
#env
date

GPU_IDS=0

CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py --dir configs/unet
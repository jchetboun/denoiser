#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=dns \
  demucs.causal=True \
  demucs.hidden=32 \
  demucs.depth=4 \
  demucs.kernel_size=4 \
  demucs.resample=1 \
  batch_size=128 \
  revecho=1 \
  segment=10 \
  stride=2 \
  shift=16000 \
  shift_same=True \
  epochs=250 \
  ddp=1

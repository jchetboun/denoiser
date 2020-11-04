#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  segment=4.5 \
  stride=0.5 \
  remix=1 \
  bandmask=0.2 \
  shift=8000 \
  shift_same=True \
  stft_loss=True \
  demucs.hidden=32 \
  demucs.causal=False \
  demucs.depth=4 \
  demucs.kernel_size=4 \
  demucs.resample=4

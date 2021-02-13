#!/usr/bin/env bash

python3 train.py \
  --max-conv-size 8 \
  --dense-kernel-size 8 \
  --num-outer-steps 2 \
  --num-inner-steps 10 \
  --force-small-data
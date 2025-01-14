#!/bin/bash

python ../sl_landscape.py -n -d "cuda" --dataset "cifar10" --distiller "../models/cifar10_norm_distill_state.ckpt" --model "../models/cifar10_norm_sl_sd.pt" "../results/cifar10_landscapes"
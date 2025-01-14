#!/bin/bash

python ../sl_landscape.py -n -d "cuda" --dataset "mnist" --distiller "../models/mnist_norm_distill_state.ckpt" --model "../models/mnist_norm_sl_sd.pt" "../results/mnist_landscapes"
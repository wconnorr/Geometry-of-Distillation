#!/bin/bash

python ../sl_landscape-evolution.py -n -d "cuda" --dataset "mnist" --model "../models/mnist_norm_sl_sd.pt" --distillers "../models/multicheckpoint" --finaldistiller "../models/multicheckpoint/checkpoint1/state.ckpt" "../results/mnist_evo_sanity"
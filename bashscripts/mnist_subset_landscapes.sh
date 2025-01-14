#!/bin/bash

python ../sl_landscape-subsets.py -n -d "cuda" --dataset "mnist" --model "../models/mnist_norm_sl_sd.pt" "../results/mnist_subset_landscapes"
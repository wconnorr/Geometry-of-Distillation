#!/bin/bash

python ../confusion_matrix.py -n -d "cpu" --distiller "../models/mnist_norm_distill_state.ckpt" --model "../models/mnist_norm_sl_sd.pt" "../results/mnist_confusion"
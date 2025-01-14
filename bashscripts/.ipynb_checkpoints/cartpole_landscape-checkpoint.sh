#!/bin/bash

python ../rl_landscape.py -d "cuda" -e "cartpole" --distiller "../models/cartpole_distiller_sd.pt" --d_critic "../models/cartpole_d_critic_sd.pt" --actor "../models/cartpole_actor_sd.pt" --critic "../models/cartpole_critic_sd.pt" --distill_zoomout --initspace_landscape "../results/cartpole_landscapes"
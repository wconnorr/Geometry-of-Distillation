#!/bin/bash

python ../rl_landscape.py -d "cuda" -e "CentipedeNoFrameskip-v4" --distiller "../models/centipede_distiller_sd.pt" --d_critic "../models/centipede_d_critic_sd.pt" --actor "../models/centipede_actor_sd.pt" --critic "../models/centipede_critic_sd.pt" --generalization_test "../results/centipede_landscapes"
#!/bin/bash

for seed in 0 1 2; do
    python extract_groups.py --dataset waterbirds --retrain --seed $seed --subset_size 0.5 --no_ddpm
    # python recipe2_end_to_end.py --dataset waterbirds --seed $seed --no_ddpm
    bash scripts/waterbirds_recipe_one.sh 0 $seed logs/waterbirds_noddpm  
done
#!/bin/bash 

python extract_groups.py --dataset waterbirds --retrain --seed 0 --subset_size 0.5
bash scripts/waterbirds_recipe_one.sh 0 0 logs/waterbirds_0
python extract_groups.py --dataset waterbirds --retrain --seed 1 --subset_size 0.5
bash scripts/waterbirds_recipe_one.sh 0 1 logs/waterbirds_1
python extract_groups.py --dataset waterbirds --retrain --seed 2 --subset_size 0.5
bash scripts/waterbirds_recipe_one.sh 0 2 logs/waterbirds_2
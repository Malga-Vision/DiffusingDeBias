#!/bin/bash 

python extract_groups.py --dataset bar --retrain --seed 0
bash scripts/bar_recipe_one.sh 0 0 logs/bar_0
python extract_groups.py --dataset bar --retrain --seed 1
bash scripts/bar_recipe_one.sh 0 1 logs/bar_1
python extract_groups.py --dataset bar --retrain --seed 2
bash scripts/bar_recipe_one.sh 0 2 logs/bar_2
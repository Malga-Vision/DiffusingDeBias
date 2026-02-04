#!/bin/bash 

python extract_groups.py --dataset bffhq --retrain --seed 0
bash scripts/bffhq_recipe_one.sh 0 0 logs/bffhq_0
python extract_groups.py --dataset bffhq --retrain --seed 1
bash scripts/bffhq_recipe_one.sh 0 1 logs/bffhq_1
python extract_groups.py --dataset bffhq --retrain --seed 2
bash scripts/bffhq_recipe_one.sh 0 2 logs/bffhq_2
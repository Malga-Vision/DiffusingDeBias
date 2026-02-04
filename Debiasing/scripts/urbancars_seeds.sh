#!/bin/bash

python extract_groups.py --dataset urbancars --retrain --seed 0
bash scripts/urbancars_recipe_one.sh 0 0 logs/urbancars_0
python extract_groups.py --dataset urbancars --retrain --seed 1
bash scripts/urbancars_recipe_one.sh 0 1 logs/urbancars_1
python extract_groups.py --dataset urbancars --retrain --seed 2
bash scripts/urbancars_recipe_one.sh 0 2 logs/urbancars_2
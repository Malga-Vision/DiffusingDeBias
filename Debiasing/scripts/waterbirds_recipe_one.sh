#!/bin/bash

python3 gdro_debiasing.py \
    -s confounder \
    -d CUB \
    -t y \
    -c ddb \
    --model resnet50 \
    --weight_decay 1.0 \
    --lr 5e-05 \
    --batch_size 128 \
    --n_epochs 60 \
    --save_step 1000 \
    --save_best \
    --save_last \
    --reweight_groups \
    --robust \
    --alpha 0.01 \
    --gamma 0.1 \
    --bias_amount $4 \
    --generalization_adjustment 2 \
    --cuda $1 \
    --seed $2 \
    --log_dir $3 
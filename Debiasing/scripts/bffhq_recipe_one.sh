#!/bin/bash

python3 gdro_debiasing.py \
    -s confounder \
    -d bffhq \
    -t y \
    -c ddb \
    --model resnet18 \
    --weight_decay 1.0 \
    --lr 5e-5 \
    --batch_size 256 \
    --n_epochs 60 \
    --save_step 1000 \
    --save_best \
    --save_last \
    --reweight_groups \
    --robust \
    --alpha 0.5 \
    --gamma 0.1 \
    --generalization_adjustment 4 \
    --cuda $1 \
    --seed $2 \
    --log_dir $3 \
    --bias_amount 99.5

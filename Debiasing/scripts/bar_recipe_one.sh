#!/bin/bash

python3 gdro_debiasing.py \
    -s confounder \
    -d BAR \
    -t action \
    -c setting \
    --model resnet18 \
    --weight_decay 0.1 \
    --lr 5e-04 \
    --batch_size 32 \
    --n_epochs 80 \
    --save_step 1000 \
    --save_best \
    --save_last \
    --robust \
    --reweight_groups \
    --alpha 0.01 \
    --gamma 0.1 \
    --cuda $1 \
    --seed $2 \
    --log_dir $3 




    
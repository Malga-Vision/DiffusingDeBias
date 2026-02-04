#!/bin/bash

python3 gdro_debiasing.py \
    -s confounder \
    -d unbiased_cifar10 \
    -t object \
    -c ddb \
    --model resnet18 \
    --weight_decay 0.01 \
    --lr 5e-04 \
    --batch_size 256 \
    --n_epochs 150 \
    --save_step 1000 \
    --save_best \
    --save_last \
    --reweight_groups \
    --robust \
    --alpha 0.01 \
    --gamma 0.1 \
    --cuda $1 \
    --seed $2 \
    --log_dir $3 \
    --bias_amount $4
import argparse
import os

import torch

from group_DRO import run_expt
from utils.wandb_wrapper import WandbWrapper

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('-d', '--dataset', choices=["CUB", "BAR", "bffhq", "urbancars", "imagenet9", "unbiased_cifar10", "bar_new", "bffhq_new"], required=True)
parser.add_argument('-s', '--shift_type', choices=["confounder", "label_shift"], required=True)
parser.add_argument('--cuda', choices=["0", "1"], required=True)

# Confounders
parser.add_argument('-t', '--target_name')
parser.add_argument('-c', '--confounder_names', nargs='+')

# Resume?
parser.add_argument('--resume', default=False, action='store_true')

# Label shifts
parser.add_argument('--minority_fraction', type=float)
parser.add_argument('--imbalance_ratio', type=float)

# Data
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--root_dir', default=None)
parser.add_argument('--reweight_groups', action='store_true', default=False)
parser.add_argument('--reweight_classes', action='store_true', default=False)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--val_fraction', type=float, default=1.0)

# Objective
parser.add_argument('--robust', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--generalization_adjustment', default="0.0")
parser.add_argument('--automatic_adjustment', default=False, action='store_true')
parser.add_argument('--robust_step_size', default=0.01, type=float)
parser.add_argument('--use_normalized_loss', default=False, action='store_true')
parser.add_argument('--btl', default=False, action='store_true')
parser.add_argument('--hinge', default=False, action='store_true')

# Model
parser.add_argument('--model', default='resnet50')
parser.add_argument('--train_from_scratch', action='store_true', default=False)
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--minimum_variational_weight', type=float, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--show_progress', default=False, action='store_true')
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--log_every', default=100, type=int)
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--save_last', action='store_true', default=False)
parser.add_argument("--bias_amount", type=float, default=0.95)
parser.add_argument("--no_wb", action="store_false", default=False, help="disables Weights & Biases logging")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f"Using Device CUDA:{torch.cuda.get_device_name()}, {torch.cuda.mem_get_info()}")
    if args.no_wb:
        wb = None
    else: wb = WandbWrapper(project_name="GDRO_Debiasing", config=args)
    run_expt.main(args, wb)


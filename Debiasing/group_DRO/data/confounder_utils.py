import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from ..models import model_attributes
from torch.utils.data import Dataset, Subset
from .celebA_dataset import CelebADataset
from .cub_dataset import CUBDataset
from .dro_dataset import DRODataset
from .bar_new import BAR as BARNew
from .imagenet9 import Imagenet9
from .urbancars import UrbanCars
from .cifar10c import CorruptedCIFAR10
from .cifar10unbiased import UnbiasedCIFAR10
from .BFFHQ_new import BFFHQ as BFFHQNew
from .bar import BAR
from .BFFHQ import BFFHQ

################
### SETTINGS ###
################

confounder_settings = {
    'celeba':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    "BAR": {
        "constructor": BAR
    },
    "bffhq": {
        "constructor": BFFHQ
    },
    "bar_new": {
        "constructor": BARNew
    },
    "bffhq_new": {
        "constructor": BFFHQNew
    },
    "cifar10c": {
        "constructor": CorruptedCIFAR10
    },
    "unbiased_cifar10": {
        "constructor": UnbiasedCIFAR10
    },
    "imagenet9": {
        "constructor": Imagenet9
    },
    "urbancars": {
        "constructor": UrbanCars
    }
    
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        bias_amount=args.bias_amount)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        if args.dataset == "BAR":
            splits = ["train", "test"]
        elif args.dataset == "imagenet9":
            splits = ["train", ]
        else:
            splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets

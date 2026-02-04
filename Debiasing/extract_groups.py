import os
from typing import List, Tuple, Union
import torch
from torch.nn.functional import one_hot
import random
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms
from matplotlib import pyplot as plt
from torch.utils import data
from datasets.cifar10c import CorruptedCIFAR10
from datasets.waterbirds import Waterbirds
from datasets.urbancars import UrbanCars
from datasets.imagenet9 import Imagenet9
from datasets.bar_new import BAR as BARNew
from datasets.BFFHQ_new import BFFHQ as BFFHQNew
from datasets.bar import BAR
from datasets.BFFHQ import BFFHQ
from datasets.waterbirds_dynamic_rho import WaterbirdsDynamicRho
from sklearn.metrics import classification_report
import pandas as pd
from utils.wandb_wrapper import WandbWrapper
import argparse


from erm_training import train_model_erm

import torch.nn as nn

class Hook:
    """Registers a hook at a specific layer of a network"""

    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module[1].register_forward_hook(self.hook_fn)
            self.name = module[0]
        else:
            self.hook = module[1].register_backward_hook(self.hook_fn)
            self.name = module[0]

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class FromNpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, num_biases=1):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.num_biases = num_biases

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        return x, tuple(y for _ in range(self.num_biases+1)), index

    def __len__(self):
        return len(self.data)

def set_seed(seed):
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ModelOnDiffused(torch.nn.Module):
    def __init__(self, num_classes, model, weight_path, dataset='waterbirds'):
        super().__init__()
        self.num_classes = num_classes
        self.model: nn.Module = copy.deepcopy(model)
        self.model.to("cuda")

        match dataset:
            case "cifar":
                self.transform = transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC)
            case _:
                self.transform = transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)

        self.model.load_state_dict(torch.load(weight_path))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        self.model.eval()
        with torch.no_grad():
            x = self.transform(x)
            return self.model(x), self.loss_fn(self.model(x), target)


def create_loaders(dataset='waterbirds', bias_amount=95, args=None):
    match dataset:
        case "urbancars":
            train_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            train_set = UrbanCars(env="train", transform=train_transform if args.no_ddpm else eval_transform)
            val_set   = UrbanCars(env="val", transform=eval_transform)
            test_set  = UrbanCars(env="test", transform=eval_transform)
            
            train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader   = data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        
        case "imagenet9":
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(TARGET_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            eval_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            train_set  = Imagenet9(env="train", transform=train_transform if args.no_ddpm else eval_transform)
            val_set, _ = data.random_split(Imagenet9(env="train", transform=eval_transform), [0.05, 0.95])
            test_set   = val_set
            
            train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
            test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        
        case 'waterbirds':
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_set = Waterbirds(root="data", env="train", transform=train_transform if args.no_ddpm else eval_transform)
            val_set = Waterbirds(root="data", env="val", transform=eval_transform)
            test_set = Waterbirds(root="data", env="test", transform=eval_transform)
            
            if args.no_ddpm:
                class_sample_count = train_set.perclass_populations()
                weight = 1.0 / class_sample_count
                targets = list()
                for i in range(0, len(train_set)):
                    targets.append(train_set.samples[i]["class_label"])
                targets = np.array(targets)
                samples_weight = np.array([weight[t] for t in targets])
                samples_weight = torch.from_numpy(samples_weight)
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=samples_weight, num_samples=len(samples_weight), replacement=True
                )
                train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, sampler=sampler, num_workers=4)
            else:
                train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, num_workers=4)
            
            val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
            test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

        case "waterbirds_dynamic":
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            train_set = WaterbirdsDynamicRho(
                root="data", 
                env="train", 
                transform=train_transform if args.no_ddpm else eval_transform, 
                bias_amount=args.rho, 
                random_seed=args.seed
            )
            
            val_set = Waterbirds(root="data", env="val", transform=eval_transform)
            test_set = Waterbirds(root="data", env="test", transform=eval_transform)
            
            if args.no_ddpm:
                class_sample_count = train_set.perclass_populations()
                weight = 1.0 / class_sample_count
                targets = list()
                for i in range(0, len(train_set)):
                    targets.append(train_set.samples[i]["class_label"])
                targets = np.array(targets)
                samples_weight = np.array([weight[t] for t in targets])
                samples_weight = torch.from_numpy(samples_weight)
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=samples_weight, num_samples=len(samples_weight), replacement=True
                )
                train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, sampler=sampler, num_workers=4)
            else:
                train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, num_workers=4)
            
            val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
            test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

        case 'cifar10c':
            eval_transform = transforms.Compose([
                transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            
            train_set    = CorruptedCIFAR10(root="data", env="train", return_index=True,  bias_amount=bias_amount, transform=eval_transform, external_bias_labels=False)
            val_set      = CorruptedCIFAR10(root="data", env="val",   return_index=True,  bias_amount=bias_amount, transform=eval_transform, external_bias_labels=False)
            test_set     = CorruptedCIFAR10(root="data", env="test",  return_index=True,  bias_amount=bias_amount, transform=eval_transform, external_bias_labels=False)
            train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader   = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
    
        case 'bar':
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_set = BAR(root="data", env="train", transform=train_transform if args.no_ddpm else eval_transform, return_index=True)
            val_set = BAR(root="data", env="val", transform=eval_transform, return_index=True)
            test_set = BAR(root="data", env="test", transform=eval_transform, return_index=True)

            train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
            test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        case 'bar_new':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_set = BARNew(root="data", env="train", transform=train_transform if args.no_ddpm else eval_transform, return_index=True, bias_amount=args.rho)
            val_set = BARNew(root="data", env="val", transform=eval_transform, return_index=True, bias_amount=args.rho)
            test_set = BARNew(root="data", env="test", transform=eval_transform, return_index=True, bias_amount=args.rho)

            train_loader = data.DataLoader(train_set, batch_size=128 if args.no_ddpm else 256, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
            test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        
        case "bffhq_new":
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_set    = BFFHQNew(root="data", env="train", transform=train_transform if args.no_ddpm else eval_transform, return_index=True, external_bias_labels=False, bias_amount=args.rho)
            val_set      = BFFHQNew(root="data", env="val", transform=eval_transform, return_index=True, external_bias_labels=False, bias_amount=args.rho)
            test_set     = BFFHQNew(root="data", env="test", transform=eval_transform, return_index=True, external_bias_labels=False, bias_amount=args.rho)
            train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader   = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        case "bffhq":
            eval_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_set    = BFFHQ(root="data", env="train", transform=train_transform if args.no_ddpm else eval_transform, return_index=True, external_bias_labels=False)
            val_set      = BFFHQ(root="data", env="val", transform=eval_transform, return_index=True, external_bias_labels=False)
            test_set     = BFFHQ(root="data", env="test", transform=eval_transform, return_index=True, external_bias_labels=False)
            train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True if args.no_ddpm else False, num_workers=4)
            val_loader   = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, eval_transform


def create_bias_amplifier(dataset_name='waterbirds'):
    match dataset_name:

        case 'waterbirds':
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)      
        
        case 'waterbirds_dynamic':
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)      
        
        case 'cifar10c':
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 10)
        
        case 'bar':
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 6)
        
        case "bffhq":
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)
            
        case 'bar_new':
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 6)
        
        case "bffhq_new":
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)
            
        case "imagenet9":
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 9)
        
        case "urbancars":
            model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)

    model_for_bias.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_for_bias = model_for_bias.to("cuda")
    return model_for_bias


def read_DDPM_images(args: argparse.Namespace):
    dataset_name = args.dataset
    norm_mode = args.norm_mode
    subset_size_pct = args.subset_size
    guidance = args.guidance_strength
    if guidance != 1:
        assert dataset_name == "waterbirds"

    guidance_folder = f"w_{guidance}"
    
    match dataset_name:
        case "urbancars":
            transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
            ])
            address = os.path.join("data", "synthetic", guidance_folder, "imagenet", "urbancars")
            files = os.listdir(address)
            x_train = list()
            y_train = list()
            for i in range(0, len(files)):
                if "npy" in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)) * int(files[i][-5]))
                    
            x_train = torch.Tensor(torch.concatenate(x_train, dim=0))
            y_train = torch.Tensor(torch.concatenate(y_train, dim=0)).long()
            dataset_DDPM = FromNpyDataset(x_train, y_train, transform, num_biases=2)
            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=256, shuffle=True, num_workers=4)
        
        case 'imagenet9':
            transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32),
            ])
            address = os.path.join("data", "synthetic", guidance_folder, "imagenet", "imagenet9")
            files = os.listdir(address)
            x_train = list()
            y_train = list()
            for i in range(0, len(files)):
                if 'npy' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)) * int(files[i][-5]))
            
            
            x_train = torch.Tensor(torch.concatenate(x_train, axis=0))
            y_train = torch.Tensor(torch.concatenate(y_train, axis=0)).long()
            dataset_DDPM = FromNpyDataset(x_train, y_train, transform)
            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=256, shuffle=True, num_workers=4)
        
        case 'cifar10c':
            transform = transforms.Compose([
                # transforms.Resize((32, 32)),
                # transforms.RandomHorizontalFlip(),

                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
            ])
            address = os.path.join("data", "synthetic", guidance_folder, norm_mode, "cifar")
            files = os.listdir(address)
            x_train = list()
            y_train = list()
            for i in range(0, len(files)):
                if 'npy' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)) * int(files[i][-5]))
            
            
            x_train = torch.Tensor(torch.concatenate(x_train, axis=0))
            y_train = torch.Tensor(torch.concatenate(y_train, axis=0)).long()
            dataset_DDPM = FromNpyDataset(x_train, y_train, transform)
            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=256, shuffle=True, num_workers=4)

        case 'bar':
            address = os.path.join("data", "synthetic", guidance_folder, norm_mode, "bar") 
            print("Loading synthetic images from ", address)
            files = os.listdir(address)
            x_train = list()
            y_train = list()
            for i in range(0, len(files)):
                if 'npy' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)) * int(files[i][-5]))

            x_train = torch.Tensor(np.concatenate(x_train, axis=0))
            y_train = torch.Tensor(np.concatenate(y_train, axis=0)).long()

            transform = transforms.Compose([  
                transforms.RandomResizedCrop(TARGET_SIZE),           
                transforms.RandomHorizontalFlip(),
            ])
            dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=128, shuffle=True, num_workers=4)

        case 'bar_new':
            address = os.path.join("data", "synthetic", guidance_folder, norm_mode, "bar_new", str(int(args.rho))) 
            print("Loading synthetic images from ", address)
            files = os.listdir(address)
            x_train = list()
            y_train = list()
            for i in range(0, len(files)):
                if 'npy' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)) * int(files[i][-5]))

            x_train = torch.Tensor(np.concatenate(x_train, axis=0))
            y_train = torch.Tensor(np.concatenate(y_train, axis=0)).long()

            transform = transforms.Compose([  
                transforms.RandomResizedCrop((64,64)),           
                transforms.RandomHorizontalFlip(),
            ])
            dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=128, shuffle=True, num_workers=4)
            
        case "waterbirds_dynamic":
            address = os.path.join("data", "synthetic", "increasing_rho", "waterbirds", str(int(args.rho)))
            print("Opening ", str(address))
            files = sorted(os.listdir(address))
            x_train = []
            y_train = []

            for i in range(0, len(files)):
                if 'npy' in files[i] and 'class_0' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.zeros(len(DDPM_images)))

                elif 'npy' in files[i] and 'class_1' in files[i]:
                    DDPM_images = np.load(os.path.join(address, files[i]))
                    x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                    y_train.append(torch.ones(len(DDPM_images)))
            
            transform = None
            dataset_DDPM = None 
            data_loader_DDPM_train = None
            
            x_train = torch.concatenate(x_train, dim=0)
            y_train = torch.concatenate(y_train, dim=0).long()
            transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])

            dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
            if subset_size_pct is not None:
                dataset_DDPM, _ = data.random_split(dataset_DDPM, [subset_size_pct, 1 - subset_size_pct])
                print(f"Using subset ({subset_size_pct}) of synthetic images, current len: ", len(dataset_DDPM))

            data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=64, shuffle=True, num_workers=4)

    if dataset_name in {"bffhq", "waterbirds", "celeba", "bffhq_new"}:
        # address = os.path.join("data", "synthetic", guidance_folder, norm_mode, dataset_name + f"{args.ddpm_iters}")
        # address = os.path.join("data", "synthetic", guidance_folder, norm_mode, dataset_name + f"{args.ddpm_iters}")
        address = os.path.join("data", "synthetic", guidance_folder, norm_mode, f"{dataset_name}_vqvae")
        
        print("Opening ", str(address))
        files = sorted(os.listdir(address))
        x_train = []
        y_train = []

        for i in range(0, len(files)):
            if 'npy' in files[i] and 'class_0' in files[i]:
                DDPM_images = np.load(os.path.join(address, files[i]))
                # x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
                x_train.append(torch.from_numpy(DDPM_images))
                y_train.append(torch.zeros(len(DDPM_images)))

            elif 'npy' in files[i] and 'class_1' in files[i]:
                DDPM_images = np.load(os.path.join(address, files[i]))
                x_train.append(torch.from_numpy(DDPM_images))
                y_train.append(torch.ones(len(DDPM_images)))
        
        transform = None
        dataset_DDPM = None 
        data_loader_DDPM_train = None
        match dataset_name:
            case "bffhq":            
                x_train = torch.concatenate(x_train, dim=0)
                y_train = torch.concatenate(y_train, dim=0).long()
                transform = transforms.Compose([
                    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                ])
                dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
                data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=32, shuffle=True, num_workers=4)
            
            case "waterbirds":
                x_train = torch.concatenate(x_train, dim=0)
                y_train = torch.concatenate(y_train, dim=0).long()
                transform = transforms.Compose([
                    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                ])

                dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
                if subset_size_pct is not None:
                    dataset_DDPM, _ = data.random_split(dataset_DDPM, [subset_size_pct, 1 - subset_size_pct])
                    print(f"Using subset ({subset_size_pct}) of synthetic images, current len: ", len(dataset_DDPM))

                data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=64, shuffle=True, num_workers=4)
            case _:
                x_train = torch.concatenate(x_train, dim=0)
                y_train = torch.concatenate(y_train, dim=0).long()

                transform = transforms.Compose([
                    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                ])

                dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
                data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=256, shuffle=True, num_workers=4)

    return dataset_DDPM, data_loader_DDPM_train


@torch.no_grad()
def extract_misclassified(dataset_name: str, biased_model: nn.Module, dataset: data.Dataset, device="cuda", rho = 95, wb: WandbWrapper = None):
    dataset: Union[BAR, Waterbirds, BFFHQ, CorruptedCIFAR10]
    class_labels: List[torch.Tensor] = []
    bias_labels: List[torch.Tensor] = []
    class_predictions: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []

    dataset.transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print(dataset.transform)

    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    x: torch.Tensor  #type-annot
    y: torch.Tensor  #type-annot
    b: torch.Tensor  #type-annot

    biased_model.eval()
    for (x, (labels), _) in dataloader:
        x = x.to(device)
        y = labels[0].to(device)
        b = ((labels[1].to(device) != y) | (labels[2].to(device) != y)).long() if dataset_name == "urbancars" else labels[1].to(device).long()

        outs: Tuple[torch.Tensor, torch.Tensor] = biased_model(x, y)
        y_preds: torch.Tensor = outs[0].argmax(dim=1)

        class_labels.append(y)
        bias_labels.append(b)
        class_predictions.append(y_preds)
        losses.append(outs[1])

    class_labels: torch.Tensor = torch.cat(class_labels, dim=0).cpu()
    bias_labels: torch.Tensor = torch.cat(bias_labels, dim=0).cpu()
    class_predictions: torch.Tensor = torch.cat(class_predictions, dim=0).cpu()
    bias_predictions: torch.Tensor = class_predictions.clone()
    losses: torch.Tensor = torch.cat(losses, dim=0).cpu()

    threshold = torch.mean(losses) + 3*torch.std(losses)
    print(classification_report(class_labels, class_predictions))

    outpath = os.path.join("outputs", dataset_name)
    os.makedirs(outpath, exist_ok=True)
    torch.save({
        "class_labels": class_labels,
        "bias_labels": bias_labels,
        "class_predictions": class_predictions,
        "bias_predictions": bias_predictions,
    }, os.path.join(outpath, f"{dataset_name}_{args.rho}_biased_model_predictions.pt"))


    new_metadata_path = None
    match dataset_name:
        case "urbancars":
            bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
            # bias_predictions = torch.where((class_predictions != class_labels), 1-class_labels, class_labels)
            
            df = pd.read_csv("./data/urbancars_images/bg-0.95_co_occur_obj-0.95/train/urbancars_metadata.csv", header="infer",
                             index_col=0)
            df["ddb"] = bias_predictions.numpy()
            new_metadata_path = os.path.join("outputs", "urbancars_metadata_aug.csv")
            df.to_csv(new_metadata_path)
            
            csv = pd.read_csv(new_metadata_path, header="infer")
            
            
            true_aligned            = csv[(csv["target"] == csv["bg_label"]) & (csv["target"] == csv["coObj_label"])]
            true_bg_conflicting     = csv[(csv["target"] != csv["bg_label"]) & (csv["target"] == csv["coObj_label"])]
            true_co_conflicting     = csv[(csv["target"] == csv["bg_label"]) & (csv["target"] != csv["coObj_label"])]
            true_bgco_conflicting   = csv[(csv["target"] != csv["bg_label"]) & (csv["target"] != csv["coObj_label"])]

            class_0_metrics_coObj = classification_report(csv[(csv["target"] == 0) & (csv["bg_label"] == 0)]["coObj_label"],
                                                    csv[(csv["target"] == 0) & (csv["bg_label"] == 0)]["ddb"],
                                                    target_names=["coObj-Aligned", "coObj-Conflicting"])
            
            class_0_metrics_bg = classification_report(csv[(csv["target"] == 0) & (csv["coObj_label"] == 0)]["bg_label"],
                                                    csv[(csv["target"] == 0) & (csv["coObj_label"] == 0)]["ddb"],
                                                    target_names=["bg-Aligned", "bg-Conflicting"])
            
            class_0_metrics_bgcoObj = classification_report(csv[(csv["target"] == 0) & (csv["coObj_label"] == 1)]["bg_label"],
                                                    csv[(csv["target"] == 0) & (csv["coObj_label"] == 1)]["ddb"],
                                                    target_names=["bg-coObj-Aligned", "bg-coObj-Conflicting"])
            
            class_1_metrics_coObj = classification_report(1-csv[(csv["target"] == 1) & (csv["bg_label"] == 1)]["coObj_label"],
                                                    1-csv[(csv["target"] == 1) & (csv["bg_label"] == 1)]["ddb"],
                                                    target_names=["coObj-Aligned", "coObj-Conflicting"])
            
            class_1_metrics_bg = classification_report(1-csv[(csv["target"] == 1) & (csv["coObj_label"] == 1)]["bg_label"],
                                                    1-csv[(csv["target"] == 1) & (csv["coObj_label"] == 1)]["ddb"],
                                                    target_names=["bg-Aligned", "bg-Conflicting"])
            
            class_1_metrics_bgcoObj = classification_report(1-csv[(csv["target"] == 1) & (csv["coObj_label"] == 0)]["bg_label"],
                                                    1-csv[(csv["target"] == 1) & (csv["coObj_label"] == 0)]["ddb"],
                                                    target_names=["bg-coObj-Aligned", "bg-coObj-Conflicting"])

            print("Class 0 (Urban)")
            print(class_0_metrics_coObj, "\n")
            print(class_0_metrics_bg, "\n")
            print(class_0_metrics_bgcoObj, "\n")
            print("Class 1 (Country)")
            print(class_1_metrics_coObj, "\n")
            print(class_1_metrics_bg, "\n")
            print(class_1_metrics_bgcoObj, "\n")
            # print("Bias-Conflicting", "\n")
            
            # print(true_bg_conflicting, "\n")
            # print(true_co_conflicting, "\n")
            # print(true_bgco_conflicting, "\n")
        
        case "imagenet9":
            bias_predictions = ((class_predictions != class_labels) & (losses > threshold)).long().numpy()
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(torch.from_numpy(bias_predictions), return_counts=True))
            for c in range(9):
                print("Class ", c)
                print(classification_report(bias_labels[class_labels == c], bias_predictions[class_labels == c]))
            new_metadata_path = os.path.join("outputs", f"imagenet9_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)
        
        case "waterbirds":
            bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
            df = pd.read_csv("./data/waterbirds/waterbird_complete95_forest2water2/metadata.csv", header="infer",
                             index_col=0)
            train_ids = df[df["split"] == 0].index.to_numpy()
            df["ddb"] = df["place"].copy()
            df.loc[train_ids, "ddb"] = bias_predictions.numpy()
            new_metadata_path = os.path.join("outputs", "waterbirds_metadata_aug.csv")
            df.to_csv(new_metadata_path)

            csv = pd.read_csv(new_metadata_path, header="infer")
            train_set = csv[csv["split"] == 0]
            true_aligned = train_set[train_set["y"] == train_set["place"]]
            true_conflicting = train_set[train_set["y"] != train_set["place"]]
            
            class_0_metrics = classification_report(train_set[train_set["y"] == 0]["place"],
                                                    train_set[train_set["y"] == 0]["ddb"],
                                                    target_names=["Aligned", "Conflicting"])
            class_1_metrics = classification_report(train_set[train_set["y"] == 1]["place"],
                                                    train_set[train_set["y"] == 1]["ddb"],
                                                    target_names=["Conflicting", "Aligned"])
            aligned_metrics = classification_report(true_aligned["place"], true_aligned["ddb"],
                                                    target_names=["Landbird", "Waterbird"])
            conflic_metrics = classification_report(true_conflicting["place"], true_conflicting["ddb"],
                                                    target_names=["Landbird", "Waterbird"])
            overall_metrics = classification_report(train_set["place"], train_set["ddb"],
                                                    target_names=["Landbird", "Waterbird"])

            print("Class 0 (Landbird)")
            print(class_0_metrics, "\n")
            print("Class 1 (Waterbird)")
            print(class_1_metrics, "\n")
            print("Bias-Conflicting", "\n")
            print(conflic_metrics, "\n")
            
        case "waterbirds_dynamic":
            bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
            # df = pd.read_csv("./data/waterbirds/waterbird_complete95_forest2water2/metadata.csv", header="infer",
            #                  index_col=0)
            df = pd.read_csv(os.path.join(dataset.data_dir, f"waterbirds_{rho}_dynamic_metadata.csv"))
            train_ids = df[df["split"] == 0].index.to_numpy()
            df["ddb"] = df["place"].copy()
            df.loc[train_ids, "ddb"] = bias_predictions.numpy()
            new_metadata_path = os.path.join("outputs", f"waterbirds_{rho}_dynamic_metadata.csv")
            df.to_csv(new_metadata_path)

            csv = pd.read_csv(new_metadata_path, header="infer")
            train_set = csv[csv["split"] == 0]
            true_aligned = train_set[train_set["y"] == train_set["place"]]
            true_conflicting = train_set[train_set["y"] != train_set["place"]]
            
            class_0_metrics = classification_report(train_set[train_set["y"] == 0]["place"],
                                                    train_set[train_set["y"] == 0]["ddb"],
                                                    target_names=["Aligned", "Conflicting"])
            class_1_metrics = classification_report(train_set[train_set["y"] == 1]["place"],
                                                    train_set[train_set["y"] == 1]["ddb"],
                                                    target_names=["Conflicting", "Aligned"])
            aligned_metrics = classification_report(true_aligned["place"], true_aligned["ddb"],
                                                    target_names=["Landbird", "Waterbird"])
            conflic_metrics = classification_report(true_conflicting["place"], true_conflicting["ddb"],
                                                    target_names=["Landbird", "Waterbird"])
            overall_metrics = classification_report(train_set["place"], train_set["ddb"],
                                                    target_names=["Landbird", "Waterbird"])

            print("Class 0 (Landbird)")
            print(class_0_metrics, "\n")
            print("Class 1 (Waterbird)")
            print(class_1_metrics, "\n")
            print("Bias-Conflicting", "\n")
            print(conflic_metrics, "\n")

        case "celeba":
            split_df = pd.read_csv("./data/CelebA/list_eval_partition.csv")
            df = pd.read_csv("./data/CelebA/list_attr_celeba.csv", sep=" ").replace(-1, 0)
            train_ids = split_df[split_df["split"] == 0].index.to_numpy()
            df["ddb"] = df["Male"].copy()
            df["split"] = split_df["split"].copy()
            df.loc[train_ids, "ddb"] = bias_predictions.numpy()
            new_metadata_path = os.path.join("outputs", "celeba_metadata_aug.csv")
            df.to_csv(new_metadata_path)

            csv = pd.read_csv(new_metadata_path, header="infer")
            train_set = csv[csv["split"] == 0]

            aligned     = train_set[train_set["Blond_Hair"] == train_set["Male"]]
            conflicting = train_set[train_set["Blond_Hair"] != train_set["Male"]] 

            class_0_metrics = classification_report(train_set[train_set["Blond_Hair"]==0]["Male"], train_set[train_set["Blond_Hair"]==0]["ddb"], target_names=["Aligned", "Conflicting"])
            class_1_metrics = classification_report(train_set[train_set["Blond_Hair"]==1]["Male"], train_set[train_set["Blond_Hair"]==1]["ddb"], target_names=["Conflicting", "Aligned"])
            aligned_metrics = classification_report(aligned["Male"], aligned["ddb"], target_names=["Not Blond", "Blond"])
            conflic_metrics = classification_report(conflicting["Male"], conflicting["ddb"], target_names=["Not Blond", "Blond"])
            overall_metrics = classification_report(train_set["Male"], train_set["ddb"], target_names=["Not Blond", "Blond"])

            print("Class 0 (Non Blond)")
            print(class_0_metrics, "\n")
            print("Class 1 (Blond)")
            print(class_1_metrics, "\n")
            print("Bias-Conflicting", "\n")
            print(conflic_metrics, "\n")

        case "bar":
            # bias_predictions = (class_predictions != class_labels).long().numpy()
            bias_predictions = ((class_predictions != class_labels) & (losses > threshold)).long().numpy()
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(torch.from_numpy(bias_predictions), return_counts=True))
            new_metadata_path = os.path.join("outputs", "bar_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)
        
        case "bar_new":
            # bias_predictions = (class_predictions != class_labels).long().numpy()
            bias_predictions = ((class_predictions != class_labels) & (losses > threshold)).long().numpy()
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(torch.from_numpy(bias_predictions), return_counts=True))
            new_metadata_path = os.path.join("outputs", f"bar_new_{args.rho}_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)

        case "cifar10c":
            bias_predictions = ((class_predictions != class_labels) & (losses > threshold)).long().numpy()
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(torch.from_numpy(bias_predictions), return_counts=True))
            for c in range(10):
                print("Class ", c)
                print(classification_report(bias_labels[class_labels == c], bias_predictions[class_labels == c]))
            new_metadata_path = os.path.join("outputs", f"cifar10c_{rho}_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)

        case "bffhq":
            bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(bias_predictions, return_counts=True))
            new_metadata_path = os.path.join("outputs", "bffhq_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)

            csv = pd.read_csv(new_metadata_path, header="infer")
            true_aligned = bias_labels[bias_labels == class_labels]
            true_conflicting = bias_labels[bias_labels != class_labels]
            pred_aligned = bias_predictions[bias_labels == class_labels]
            pred_conflicting = bias_predictions[bias_labels != class_labels]

            class_0_metrics = classification_report(bias_labels[class_labels == 0], bias_predictions[class_labels == 0],
                                                    target_names=["Aligned", "Conflicting"])
            class_1_metrics = classification_report(bias_labels[class_labels == 1], bias_predictions[class_labels == 1],
                                                    target_names=["Conflicting", "Aligned"])
            aligned_metrics = classification_report(true_aligned, pred_aligned, target_names=["Young", "Old"])
            conflic_metrics = classification_report(true_conflicting, pred_conflicting, target_names=["Young", "Old"])
            overall_metrics = classification_report(bias_labels, bias_predictions, target_names=["Young", "Old"])

            print("Class 0 (Young)")
            print(class_0_metrics, "\n")
            print("Class 1 (Old)")
            print(class_1_metrics, "\n")
            print("Bias-Conflicting", "\n")
            print(conflic_metrics, "\n")

            if wb is not None:
                wb.log_output({
                    "id_metrics": 
                        wb.backend.Table(dataframe=pd.DataFrame.from_dict({ 
                        "class_0": class_0_metrics,
                        "class_1": class_1_metrics,
                        "aligned": aligned_metrics,
                        "conflic": conflic_metrics,
                        "overall": overall_metrics
                    }, orient="index"))
                })
                metadata_file = wb.backend.Artifact("pseudolabels_metadata_file", type="dataset")
                metadata_file.add_file(new_metadata_path)
                wb.backend.log_artifact(metadata_file)
        
        case "bffhq_new":
            bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
            pred_csv = pd.DataFrame(bias_predictions, columns=["ddb", ])
            print(torch.unique(bias_predictions, return_counts=True))
            new_metadata_path = os.path.join("outputs", f"bffhq_new_{args.rho}_metadata_aug.csv")
            pred_csv.to_csv(new_metadata_path)

            csv = pd.read_csv(new_metadata_path, header="infer")
            true_aligned = bias_labels[bias_labels == class_labels]
            true_conflicting = bias_labels[bias_labels != class_labels]
            pred_aligned = bias_predictions[bias_labels == class_labels]
            pred_conflicting = bias_predictions[bias_labels != class_labels]

            class_0_metrics = classification_report(bias_labels[class_labels == 0], bias_predictions[class_labels == 0],
                                                    target_names=["Aligned", "Conflicting"])
            class_1_metrics = classification_report(bias_labels[class_labels == 1], bias_predictions[class_labels == 1],
                                                    target_names=["Conflicting", "Aligned"])
            aligned_metrics = classification_report(true_aligned, pred_aligned, target_names=["Young", "Old"])
            conflic_metrics = classification_report(true_conflicting, pred_conflicting, target_names=["Young", "Old"])
            overall_metrics = classification_report(bias_labels, bias_predictions, target_names=["Young", "Old"])

            print("Class 0 (Young)")
            print(class_0_metrics, "\n")
            print("Class 1 (Old)")
            print(class_1_metrics, "\n")
            print("Bias-Conflicting", "\n")
            print(conflic_metrics, "\n")

            if wb is not None:
                wb.log_output({
                    "id_metrics": 
                        wb.backend.Table(dataframe=pd.DataFrame.from_dict({ 
                        "class_0": class_0_metrics,
                        "class_1": class_1_metrics,
                        "aligned": aligned_metrics,
                        "conflic": conflic_metrics,
                        "overall": overall_metrics
                    }, orient="index"))
                })
                metadata_file = wb.backend.Artifact("pseudolabels_metadata_file", type="dataset")
                metadata_file.add_file(new_metadata_path)
                wb.backend.log_artifact(metadata_file)

    return bias_predictions


def extract_groups(args: argparse.Namespace):
    dataset_name  = args.dataset
    retrain_model = args.retrain
    norm_mode     = args.norm_mode
    use_wb        = not args.no_wb 
    rho           = args.rho
    subset_size_pct = args.subset_size
    no_ddpm = args.no_ddpm

    if dataset_name == "cifar10c":
        global TARGET_SIZE
        TARGET_SIZE = (32, 32)
    elif dataset_name == "imagenet9":
        TARGET_SIZE = (224, 224)    
    if no_ddpm: 
        TARGET_SIZE = (224, 224)
    
    train_loader, _, _, eval_transform = create_loaders(dataset_name, rho, args)
    config = {"dataset_name": dataset_name, }
    match dataset_name:
        case "urbancars":
            train_set = UrbanCars(env="train", return_index=True)
            config["lr"] = 1e-4
            config["num_classes"] = 2
            config["epochs"] = 50
            config["wd"] = 0.01
        case "imagenet9":
            train_set = Imagenet9(env="train", return_index=True)
            config["lr"] = 0.0005
            config["num_classes"] = 9
            config["epochs"] = 20
            config["wd"] = 0.01
        case "waterbirds":
            train_set = Waterbirds(env="train", return_index=True)
            config["lr"] = 0.0005
            config["num_classes"] = 2
            config["epochs"] = 100
            config["wd"] = 0.01
        case "waterbirds_dynamic":
            train_set = WaterbirdsDynamicRho(
                env="train", 
                bias_amount=args.rho, 
                random_seed=args.seed, 
                metadata_path=f"./data/waterbirds/waterbird_complete95_forest2water2/waterbirds_{float(args.rho)}_dynamic_metadata.csv"
            )
            config["lr"] = 0.0005
            config["num_classes"] = 2
            config["epochs"] = 50
            config["wd"] = 0.01
        case "bar":
            train_set = BAR(env="train", return_index=True, transform=eval_transform)
            config["lr"] = 0.0005
            config["num_classes"] = 6
            config["epochs"] = 50
            config["wd"] = 0.01
        case "bffhq":
            train_set = BFFHQ(env="train", return_index=True, transform=eval_transform, external_bias_labels=False)
            config["lr"] = 0.0005
            config["num_classes"] = 2
            config["epochs"] = 100
            config["wd"] = 0.01
        case "bar_new":
            # train_set = BARNew(env="train", return_index=True, transform=eval_transform, bias_amount=rho)
            # config["lr"] = 0.0005
            # config["num_classes"] = 6
            # config["epochs"] = 50
            # config["wd"] = 0.01
            train_set = BARNew(env="train", return_index=True, transform=eval_transform, bias_amount=rho)
            config["lr"] = 0.0005
            config["num_classes"] = 6
            config["epochs"] = 30
            config["wd"] = 0.01
        case "bbfhq_new":
            train_set = BFFHQNew(env="train", return_index=True, transform=eval_transform, external_bias_labels=False, rho=rho)
            config["lr"] = 0.0005
            config["num_classes"] = 2
            config["epochs"] = 5#100
            config["wd"] = 1.0#0.01
        case "cifar10c":
            train_set = CorruptedCIFAR10(
                env="train", 
                bias_amount=rho,
                return_index=True, 
                transform=eval_transform, 
                external_bias_labels=False
            )
            config["lr"] = 0.0005
            config["num_classes"] = 10
            config["epochs"] = 50
            config["wd"] = 0.01
        case _:
            raise ValueError(f"'{dataset_name}' unsupported dataset name")

    biased_model = create_bias_amplifier(dataset_name)
    if no_ddpm:
        print("Using original training set for training the BiasAmplifier")
        data_loader_DDPM_train = train_loader
    else:
        _, data_loader_DDPM_train = read_DDPM_images(args)

    wb = None
    if use_wb:
        wb = WandbWrapper("DiffuseDebias_0", config)

    print(train_set)
    if retrain_model:
        train_model_erm(
            biased_model,
            data_loader_DDPM_train,
            None,
            None,
            "cuda",
            torch.optim.AdamW(biased_model.parameters(), lr=config["lr"], weight_decay=config["wd"]),
            num_classes=config["num_classes"],
            num_biases = 2 if dataset_name == "urbancars" else 1,
            epochs=config["epochs"],
            wb=wb,
            name=f"biased_model_{rho}_{dataset_name}" + ("_no_ddpm" if args.no_ddpm else "")
        )

    biased_model = ModelOnDiffused(
        config["num_classes"],
        biased_model,
        os.path.join(
            PATH_TO_MODELS, 
            f"biased_model_{rho}_{dataset_name}" + ("_no_ddpm" if args.no_ddpm else "") + "-final.pt"
        ),
        dataset=dataset_name
    )

    bias_pseudolabels: torch.Tensor = extract_misclassified(dataset_name, biased_model, train_set, "cuda", rho=rho, wb=wb)
    print(torch.unique(torch.as_tensor(bias_pseudolabels), return_counts=True))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="waterbirds", required=True, help="dataset name. choose in [waterbirds, bar, bffhq, celeba, cifar10c]")
parser.add_argument("--retrain", action="store_true", help="repeat experiment and overwrite biased model, default=False")
parser.add_argument("--no_ddpm", action="store_true")
parser.add_argument("--no_wb", action="store_false", default=False, help="disables Weights & Biases logging")
parser.add_argument("--ddpm_iters", type=str, default="")
parser.add_argument("--norm_mode", type=str, default="imagenet", help="Normalization protocol for generated images. choose in [imagenet, minmax]. Default: imagenet")
parser.add_argument("--seed", type=int, default=0, help="random state for stochastic operations")
parser.add_argument("--rho", type=float, default=95)
parser.add_argument("--subset_size", type=float, default=None, help="Subset size of synthetic images for training the biased model")
parser.add_argument("--guidance_strength", type=int, default=1, help="Ablation study on classifier strength 'w'. Default=1, choose among [0, 1, 2, 3, 5]")

if __name__ == "__main__":   
    PATH_TO_MODELS = f'saved_models'
    TARGET_SIZE = (64, 64)
    args = parser.parse_args()
    set_seed(args.seed)   
    
    if args.no_ddpm:
        TARGET_SIZE = (224, 224) 
    
    print(args)
    extract_groups(args)

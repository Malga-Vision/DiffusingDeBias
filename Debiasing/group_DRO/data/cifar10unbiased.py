from typing import Generator, Tuple, Union
import gdown
import tarfile
import os

from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import torchvision
import pandas as pd
import torch


class UnbiasedCIFAR10(Dataset):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    def __init__(
            self, 
            root_dir="./data", 
            env="train", 
            bias_amount = 0.1, 
            target_name = "object", 
            confounder_names="object", 
            return_index = False, 
            transform=None, 
            image_path_list=None,
            **kwargs
        ):
        super().__init__()

        if transform is None:
            self.transform = UnbiasedCIFAR10.train_transform if env == "train" else UnbiasedCIFAR10.eval_transform
        else: 
            self.transform = transform
        
        
        self.dataset = torchvision.datasets.CIFAR10(root="./data", train=(env == "train"), transform=self.transform, download=True)
        
        self.env = env
        self.root_dir = root_dir
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.confounder_array = None
        self.n_classes = 10
        self.n_groups = 20
        self.n_confounders = 1
        self.return_index = return_index
        self.bias_amount = bias_amount
        self.y_array = torch.as_tensor(self.dataset.targets)
        
        if self.env == "train":            
            print("Loading external bias labels!")
            self.confounder_array = pd.read_csv(os.path.join("outputs", f"cifar10_unbiased_metadata_aug.csv"))["ddb"].to_numpy()
            print("Loaded ", f"cifar10_unbiased_metadata_aug.csv")
        else:
            self.confounder_array = torch.zeros(len(self.dataset))
            print("External bias labels are supported only for the training set, skipping...")

        self.group_array = ((self.y_array * 2) + self.confounder_array).long()
        
    def __getitem__(self, index):        
        bias_label = self.group_array[index]
        x, y = self.dataset[index]
        
        return x, y, bias_label
    
    def __len__(self):
        return len(self.dataset)
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][1][1]

    def get_class_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][1][0]

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), f"{split} is not a valid split"
            split_set = UnbiasedCIFAR10(env=split, return_index=True, bias_amount=self.bias_amount)
            subsets[split] = split_set

        return subsets
    
    def group_str(self, group_idx):
        # Calculate the class (y) and confounder (c) for the group index
        y = group_idx // (self.n_groups // self.n_classes)  # Class label
        c = group_idx % (self.n_groups // self.n_classes)   # Confounder

        # Combine class and confounder into the group name
        group_name = f"Class: {int(y)}, Confounder: {c}"
        
        return group_name
    
    def get_group_array(self):
        return self.group_array
    
    def get_label_array(self):
        return self.y_array

if __name__ == '__main__':
    pass

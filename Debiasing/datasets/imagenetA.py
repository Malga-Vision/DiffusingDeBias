#!/usr/bin/env python3


# https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import List, Callable, Tuple, Generator, Union
from collections import OrderedDict
from torch.utils.data import ConcatDataset
import pandas as pd
import requests
from tqdm import tqdm
import tarfile   

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class ImageNetA(Dataset):
    DOWNLOAD_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    DATASET_NAME = "imagenet-a"

    def __init__(self, root: str = "./data", transform = data_transform, external_cls_to_idx=None):
        self.root:              str  = os.path.join(root, ImageNetA.DATASET_NAME)
        self.transform = transform
        self.external_cls_to_idx = external_cls_to_idx

        if not os.path.isdir(os.path.join(self.root, "imagenet-a")):
            self.__download_dataset()
        else: self.root = os.path.join(self.root, "imagenet-a")

        self.dataset: ImageFolder = ImageFolder(
            self.root, 
            transform=data_transform, 
            target_transform=transform, 
            is_valid_file=lambda filename : any(map(lambda ext : filename.lower().endswith(ext), [".jpg", ".jpeg", ".png", ".tif"]))
        )

        self.num_classes = len(self.dataset.classes)

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "imagenet-a.tar")
        print(f"=> Downloading {ImageNetA.DATASET_NAME} for {ImageNetA.DOWNLOAD_URL}")

        try:
            response = requests.get(ImageNetA.DOWNLOAD_URL, stream=True)
            response.raise_for_status()

            with open(output_path, mode="wb") as write_stream, tqdm(
                desc=output_path,
                total=int(response.headers["content-length"], 0),
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    write_stream.write(chunk)
                    pbar.update(len(chunk))

        except:
            raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
        print(f"=> Extracting imagenet-a.tar to directory {self.root}")
        try:
            with tarfile.open(output_path, mode="r:tar") as unballer:
                unballer.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.root = os.path.join(self.root, "imagenet-a")
        os.remove(output_path)        


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image_path, class_label = self.dataset.samples[index]
        if self.external_cls_to_idx is not None:
            class_label = self.external_cls_to_idx[self.dataset.classes[class_label]]
        image = self.transform(Image.open(image_path).convert("RGB"))
        
        return  image, (class_label, class_label, ), index 
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts

    def __repr__(self) -> str:
        return f"ImageNet-A(num_classes={self.num_classes})"
    

if __name__ == "__main__":
    
    ImageNetA(env="test")
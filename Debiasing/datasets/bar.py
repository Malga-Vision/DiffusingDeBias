#!/usr/bin/env python3

"https://github.com/alinlab/BAR/archive/refs/heads/master.zip"

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Generator, Union
import os
import shutil
import sys
from os import path
from PIL import Image
import requests
from tqdm import tqdm
import zipfile

class BAR(Dataset):
    DOWNLOAD_URL = "https://api.github.com/repos/alinlab/BAR/zipball"
    DATASET_NAME = "bar"
    
    eval_transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    classes_str_to_idx = {
        "climbing": 0,
        "diving":   1,
        "fishing":  2,
        "racing":   3,
        "throwing": 4,
        "pole vaulting": 5,
    }

    def __init__(self, root="./data", env="train", return_index=False, transform=None) -> None:
        self.root = root
        if transform is None:
            self.transform = BAR.eval_transform if env == "test" else BAR.train_transform
        else: self.transform = transform
        
        # TODO: Better control the usage of envs
        if env == "val":
            self.env = "train" 
        else:
            self.env = env 
        
        self.return_index = return_index
        self.num_classes = 6

        if not os.path.isdir(os.path.join(self.root, "alinlab-BAR-1364b0a")):
            self.__download_dataset()
        else: self.root = os.path.join(self.root, "alinlab-BAR-1364b0a")

        self.samples = {}
        self.num_samples = 0

        self.samples_folder = path.join(self.root, self.env)

        for i, file in enumerate(sorted(os.listdir(self.samples_folder))):
            self.samples[i] = {
                "image_path": path.join(self.samples_folder, file),
                "class_label": int(BAR.classes_str_to_idx[file.split("_")[0]]),
                "bias_label": int(BAR.classes_str_to_idx[file.split("_")[0]]),
            }
            self.num_samples += 1

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]
        
        image = self.transform(Image.open(self.samples[index]["image_path"]).convert("RGB"))
        class_label = self.samples[index]["class_label"]
        bias_label = self.samples[index]["bias_label"]


        return image, (class_label, bias_label, ), index

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "BAR-master.zip")
        print(f"=> Downloading {BAR.DATASET_NAME} for {BAR.DOWNLOAD_URL}")

        try:
            response = requests.get(BAR.DOWNLOAD_URL, stream=True, headers={'Accept-Encoding': None})
            response.raise_for_status()
            with open(output_path, mode="wb") as write_stream:
                for chunk in response.iter_content(chunk_size=8192):
                    write_stream.write(chunk)                  

        except:
            raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
        print(f"=> Extracting BAR-master.zip to directory {self.root}")
        try:
            with zipfile.ZipFile(output_path, mode="r") as unzipper:
                unzipper.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")
        
        self.root = os.path.join(self.root, "alinlab-BAR-1364b0a")
        os.remove(output_path)
    
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
            _, (y, b), idx = self[i]
            yield b

    def get_class_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            _, (y, b), idx = self[i]
            yield y

    
    def __repr__(self) -> str:
        return f"BAR(env={self.env}, bias_amount=Unknown, num_classes={self.num_classes})"


if __name__ == "__main__":    
    d = BAR()
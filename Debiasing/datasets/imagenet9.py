#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Callable, Tuple, Generator, Union
from collections import OrderedDict
from torch.utils.data import ConcatDataset
import pandas as pd
import requests
from tqdm import tqdm
import tarfile
    


data_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class Imagenet9(Dataset):
    DOWNLOAD_URL = ""
    DATASET_NAME = "imagenet9"
    
    CLASS_TO_INDEX = {'n01641577': 2, 'n01644373': 2, 'n01644900': 2, 'n01664065': 3, 'n01665541': 3,
                  'n01667114': 3, 'n01667778': 3, 'n01669191': 3, 'n01819313': 4, 'n01820546': 4,
                  'n01833805': 4, 'n01843383': 4, 'n01847000': 4, 'n01978287': 7, 'n01978455': 7,
                  'n01980166': 7, 'n01981276': 7, 'n02085620': 0, 'n02099601': 0, 'n02106550': 0,
                  'n02106662': 0, 'n02110958': 0, 'n02123045': 1, 'n02123159': 1, 'n02123394': 1,
                  'n02123597': 1, 'n02124075': 1, 'n02174001': 8, 'n02177972': 8, 'n02190166': 8,
                  'n02206856': 8, 'n02219486': 8, 'n02486410': 5, 'n02487347': 5, 'n02488291': 5,
                  'n02488702': 5, 'n02492035': 5, 'n02607072': 6, 'n02640242': 6, 'n02641379': 6,
                  'n02643566': 6, 'n02655020': 6}

    def __init__(self, env: str, root: str = "./data", transform = data_transform, metadata_filename: str = "imagenet_9.csv", return_index: bool = True):
        self.root:              str  = os.path.join(root, Imagenet9.DATASET_NAME)
        self.env:               str  = env
        self.metadata_filename: str  = metadata_filename
        self.return_index:      bool = return_index
        self.num_classes = 9

        assert (env == "train")

        if "imagenet9" not in os.listdir("./data"):
            # self.__download_dataset()
            pass
        else: self.root = "./data/imagenet9/"

        self.transform = transform
        self.metadata_path = os.path.join(self.root, self.metadata_filename)

        metadata_csv = pd.read_csv(self.metadata_path, header="infer")

        self.samples = {}
        self.files_count = 0
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            self.samples[i] = {
                "image_path":  os.path.join(self.root, sample_info["path"]),
                "class_label": int(sample_info["target"]),
                "bias_label":  0,
                "wordnetid": list((str(e) for e in sample_info))
            }
            self.files_count += 1

    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image = self.transform(Image.open(self.samples[index]["image_path"]).convert("RGB"))
        class_label = self.samples[index]["class_label"]
        bias_label  = self.samples[index]["bias_label"]
        wordnetid   = self.samples[index]["wordnetid"]

        return image, (class_label, class_label), index
    
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
        return f"imagenet9(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"
    

if __name__ == "__main__":
    Imagenet9(env="train")
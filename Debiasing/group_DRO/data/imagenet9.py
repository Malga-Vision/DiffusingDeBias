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

    def __init__(
        self, 
        env: str="train", 
        root_dir: str = "./data", 
        target_name = "object",
        confounder_names = "texture",
        transform = data_transform, 
        metadata_filename: str = "imagenet_9.csv", 
        return_index: bool = True,
        external_bias_labels: bool = True,
        **kwargs
    ):
        
        self.root:              str  = os.path.join("./data", Imagenet9.DATASET_NAME)
        self.env:               str  = env
        self.metadata_filename: str  = metadata_filename
        self.return_index:      bool = return_index
        self.num_classes = 9
        self.target_name = target_name
        self.confounder_names = confounder_names

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
        
        self.bias_labels = None
        if external_bias_labels:
            self.bias_labels = pd.read_csv(os.path.join("outputs", "imagenet9_metadata_aug.csv"), header="infer")["ddb"].to_numpy()        
        
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            self.samples[i] = {
                "image_path" : os.path.join(self.root, sample_info["path"]),
                "class_label": int(sample_info["target"]),
                "bias_label" : int(sample_info["target"]) if self.bias_labels is None else self.bias_labels[i],
                "wordnetid"  : list((str(e) for e in sample_info))
            }
            self.files_count += 1
            
        self.filename_array = np.array([self.samples[j]["image_path"]  for j in range(len(self.samples))])
        self.y_array        = np.array([self.samples[j]["class_label"] for j in range(len(self.samples))])
        self.group_array    = np.array([self.samples[j]["bias_label"]  for j in range(len(self.samples))])
        self.wordnetids     = np.array([self.samples[j]["wordnetid"]   for j in range(len(self.samples))])
        
        self.n_classes = 9
        self.n_confounders = 1
        self.n_groups = self.n_classes * 2
        self.group_array = (2 * self.y_array).astype("int")
        

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "imagenet9.tar.gz")
        print(f"=> Downloading {Imagenet9.DATASET_NAME} for {Imagenet9.DOWNLOAD_URL}")

        try:
            response = requests.get(Imagenet9.DOWNLOAD_URL, stream=True)
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
        
        print(f"=> Extracting waterbird_complete95_forest2water2.tar.gz to directory {self.root}")
        try:
            with tarfile.open(output_path, mode="r:gz") as unballer:
                unballer.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.root = os.path.join(self.root, "waterbird_complete95_forest2water2")
        os.remove(output_path)
        
    def __len__(self) -> int:
        return len(self.y_array)
    
    def get_group_array(self):
        return self.group_array

    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image = self.transform(Image.open(self.samples[index]["image_path"]).convert("RGB"))
        class_label = self.y_array[index]
        bias_label  = self.group_array[index]
        wordnetid   = self.samples[index]["wordnetid"]

        return image, class_label, bias_label
    
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

    def group_str(self, group_idx):
        # Calculate the class (y) and confounder (c) for the group index
        y = group_idx // (self.n_groups // self.n_classes)  # Class label
        c = group_idx % (self.n_groups // self.n_classes)   # Confounder

        # Convert confounder to binary string format with leading zeros (based on n_confounders)
        bin_str = format(c, f"0{self.n_confounders}b")

        # Combine class and confounder into the group name
        group_name = f"Class: {int(y)}, Confounder: {bin_str}"
        
        return group_name
    
    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "test")
            split_set = Imagenet9(env="train", return_index=True)
            subsets[split] = split_set
            
        return subsets
        

    def __repr__(self) -> str:
        return f"imagenet9(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"
    

if __name__ == "__main__":
    Imagenet9(env="train")
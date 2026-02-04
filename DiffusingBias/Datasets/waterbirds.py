#!/usr/bin/env python3
#%%
# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Subset
from collections import defaultdict
from typing import Dict

data_transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(normalization='standard'),
    A.Normalize(normalization='min_max'),
    A.pytorch.ToTensorV2()
])


class BalancedSubset(Subset):
    """Extended Subset class that maintains access to original dataset attributes"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Keep reference to the original dataset
        self.original_dataset = dataset

    @property
    def num_classes(self):
        return self.original_dataset.num_classes

def create_balanced_dataset(train_dataset, align_count: int = 642, conflict_count: int = 64):
    """
    Create a balanced dataset with specified number of aligned and conflicting samples per class.
    Works with Waterbirds dataset.
    """
    
    def get_class_bias_indices(dataset) -> Dict[int, Dict[int, List[int]]]:
        """Get indices for each class and bias combination"""
        indices = defaultdict(lambda: defaultdict(list))
        for idx in range(len(dataset)):
            sample = dataset[idx]
            indices[sample['class_label']][sample['bias_label']].append(idx)
        return indices

    # Get indices for each class and bias combination
    class_bias_indices = get_class_bias_indices(train_dataset)
    
    # Get number of classes - Waterbirds has 2 classes (0 and 1)
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Select balanced samples for each class
    selected_indices = []
    for class_label in range(num_classes):
        # Get align samples (bias_label = 1)
        align_indices = class_bias_indices[class_label][1]
        if len(align_indices) < align_count:
            print(f"Warning: Not enough aligned samples for class {class_label}. "
                  f"Available: {len(align_indices)}, Required: {align_count}")
            align_count_actual = len(align_indices)
        else:
            align_count_actual = align_count
        selected_align = torch.randperm(len(align_indices))[:align_count_actual].tolist()
        selected_indices.extend([align_indices[i] for i in selected_align])
        
        # Get conflict samples (bias_label = -1)
        conflict_indices = class_bias_indices[class_label][-1]
        if len(conflict_indices) < conflict_count:
            print(f"Warning: Not enough conflicting samples for class {class_label}. "
                  f"Available: {len(conflict_indices)}, Required: {conflict_count}")
            conflict_count_actual = len(conflict_indices)
        else:
            conflict_count_actual = conflict_count
        selected_conflict = torch.randperm(len(conflict_indices))[:conflict_count_actual].tolist()
        selected_indices.extend([conflict_indices[i] for i in selected_conflict])
    
    # Create the balanced subset using our custom BalancedSubset class
    balanced_dataset = BalancedSubset(train_dataset, selected_indices)
    
    # Print distribution statistics
    distribution = defaultdict(lambda: defaultdict(int))
    for idx in selected_indices:
        sample = train_dataset[idx]
        distribution[sample['class_label']][sample['bias_label']] += 1
    
    print("\nFinal Distribution:")
    for class_label in range(num_classes):
        print(f"\nClass {class_label}:")
        print(f"- Aligned samples: {distribution[class_label][1]}")
        print(f"- Conflicting samples: {distribution[class_label][-1]}")
    
    return balanced_dataset

class Waterbirds(Dataset):
    DOWNLOAD_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"

    def __init__(self, env: str, root: str = "./data", transform = None, metadata_filename: str = "metadata.csv", return_index: bool = False, class_label: int = None):
        self.root:              str  = root
        self.env:               str  = env
        self.metadata_filename: str  = metadata_filename
        self.return_index:      bool = return_index
        self.num_classes = 2 if class_label is None else 1

        self.env_to_split = {
            "train": 0,
            "val":   1,
            "test":  2
        }
        if not os.path.exists(os.path.join(self.root, "waterbirds")):        
            self.__download_dataset()
        
        self.root = os.path.join(self.root, "waterbirds")
        self.transform = transform
        self.metadata_path = os.path.join(self.root, self.metadata_filename)

        metadata_csv = pd.read_csv(self.metadata_path)
        metadata_csv = metadata_csv.query(f"split == {self.env_to_split[self.env]}")

        self.samples = {}
        self.files_count = 0
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            # Filter out the class label
            if class_label is not None and int(sample_info["y"]) != class_label:
                continue
            
            self.samples[self.files_count] = {
                "image_path":  os.path.join(self.root, sample_info["img_filename"]),
                "class_label": int(sample_info["y"]),
                "bias_label": -1 if int(sample_info["y"]) != int(sample_info["place"]) else 1,
                "all_attrs": list((str(e) for e in sample_info))
            }
            self.files_count += 1

    
    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "waterbird_complete95_forest2water2.tar.gz")
        print(f"=> Downloading {os.path.basename(self.root)} dataset from {self.DOWNLOAD_URL}")

        try:
            response = requests.get(Waterbirds.DOWNLOAD_URL, stream=True)
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
        # Rename the extracted folder to "waterbirds"
        os.rename(os.path.join(self.root, "waterbird_complete95_forest2water2"), os.path.join(self.root, "waterbirds"))
        os.remove(output_path)
        

    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]):
        
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        #image = self.transform(Image.open(self.samples[index]["image_path"]))
        np_image=np.array(Image.open(self.samples[index]["image_path"]))
        image = self.transform(image=np_image)["image"] 
        class_label = self.samples[index]["class_label"]
        bias_label  = self.samples[index]["bias_label"]
        
        data_dict = {
            'name': self.samples[index]["image_path"], 
            'image': image, 
            'class_label': class_label, 
            'bias_label': bias_label
        }  
        
        if self.return_index:
            data_dict["index"] = index
        
        
        return data_dict
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self.samples[i]["class_label"]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self.samples[i]["bias_label"]

    def __repr__(self) -> str:
        return f"Waterbirds(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"
    
    
if __name__ == "__main__":

    DATA_PATH = "/path/to/data"
    CLASS_LABEL = 0
    
    dataset = Waterbirds(env="train", root=DATA_PATH, transform=data_transform, class_label=CLASS_LABEL)
    print(f"Dataset: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
    print(f"Per class populations: {dataset.perclass_populations(return_labels=True)}")

    # Visualize the first 5 samples
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(5):
        sample = dataset[i]
        ax[i].imshow(sample["image"].permute(1, 2, 0))
        ax[i].set_title(f"Class: {sample['class_label']}, Bias: {sample['bias_label']}")
        ax[i].axis("off")


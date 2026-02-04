#!/usr/bin/env

import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
import requests
import zipfile

class BFFHQ(Dataset):
    DOWNLOAD_URL = "https://drive.google.com/file/d/1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp/view?usp=sharing"
    DATASET_NAME = "bffhq"

    def __init__(self, root="./data/bffhq", env="train", bias_amount=0.995, transform=None, return_index=False, class_label: int = None):
        self.root = root
        self.transform = transform
        self.env = env
        self.bias_amount=bias_amount
        self.return_index = return_index
        self.num_classes = 2 if class_label is None else 1

        self.bias_folder_dict = {
            0.995: "0.5pct"
        }

        if not os.path.isdir(os.path.join(self.root, BFFHQ.DATASET_NAME)):
            self.__download_dataset()
        else: self.data_dir = os.path.join(self.root, BFFHQ.DATASET_NAME)


        if self.env == "train":
            self.samples, self.class_labels, self.bias_labels = self.load_train_samples()

        if self.env == "val":
            self.samples, self.class_labels, self.bias_labels = self.load_val_samples()

        if self.env == "test":
            self.samples, self.class_labels, self.bias_labels = self.load_test_samples()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        class_label = self.class_labels[idx]
        bias_label = self.bias_labels[idx]

        np_image=np.array(Image.open(file_path))
        image = self.transform(image=np_image)["image"] 

        if self.return_index:
            return image, class_label, bias_label, idx

        data_dict = {
            'name': file_path, 
            'image': image, 
            'class_label': class_label, 
            'bias_label': bias_label
        }  
        
        if self.return_index:
            data_dict["index"] = idx
        
        return data_dict 
           
    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "bffhq.zip")
        print(f"=> Downloading {BFFHQ.DATASET_NAME} for {BFFHQ.DOWNLOAD_URL}")

        try:
            gdown.download(id="1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp", output=output_path)
        except:
            raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
        print(f"=> Extracting bffhq.zip to directory {self.root}")
        try:
            with zipfile.ZipFile(output_path, mode="r") as unzipper:
                unzipper.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.data_dir = os.path.join(self.root, BFFHQ.DATASET_NAME)
        os.remove(output_path)


    def load_train_samples(self, class_label=None):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]
        
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "align", class_folder))):
                
                if class_label is not None and class_label != self.assign_class_label(filename):
                    continue
                
                samples_path.append(os.path.join(self.data_dir, bias_folder, "align", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        for class_folder in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "conflict"))):
            
            if class_label is not None and class_label != self.assign_class_label(filename):
                continue
            
            for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "conflict", class_folder))):
                samples_path.append(os.path.join(self.data_dir, bias_folder, "conflict", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))     

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_val_samples(self, class_label=None):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]

        for filename in sorted(os.listdir(os.path.join(self.data_dir, bias_folder, "valid"))):
            
            if class_label is not None and class_label != self.assign_class_label(filename):
                continue
            
            samples_path.append(os.path.join(self.data_dir, bias_folder, "valid", filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_test_samples(self, class_label=None):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        for filename in sorted(os.listdir(os.path.join(self.data_dir, "test"))):
            
            if class_label is not None and class_label != self.assign_class_label(filename):
                continue
                
            samples_path.append(os.path.join(self.data_dir, "test", filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def assign_bias_label(self, filename: str) -> int:
        no_extension = filename.split(".")[0]
        _, y, z = no_extension.split("_")
        y = int(y)
        z = int(z)
        return 1 if y == z else -1
    
    def assign_class_label(self, filename: str):
        no_extension = filename.split(".")[0]
        _, y, _ = no_extension.split("_")
        return int(y)
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i]["class_label"]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i]["bias_label"]

    
    def __repr__(self) -> str:
        return f"BFFHQ(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.num_classes})"
# #!/usr/bin/env python3

# import numpy as np
# import torch
# import os
# from torchvision import transforms
# from PIL import Image
# from matplotlib import pyplot as plt
# from torch.utils.data import Dataset
# from typing import List, Callable, Tuple, Generator, Union
# import gdown
# import requests
# import zipfile
# from tqdm import tqdm
# import pandas as pd

# class BFFHQ(Dataset):
#     DOWNLOAD_URL = "https://drive.google.com/file/d/1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp/view?usp=sharing"
#     DATASET_NAME = "bffhq"    

#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#         transforms.RandomCrop(224, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     eval_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])

#     def __init__(
#             self, 
#             root_dir="./data", 
#             env="train", 
#             bias_amount=0.995,
#             target_name="age",
#             confounder_names="gender",
#             return_index = False,
#             transform=None,
#             external_bias_labels: bool = True,
#             **kwargs
#         ):
#         self.root = "./data"
#         if transform is None:
#             self.transform = BFFHQ.train_transform if env == "train" else BFFHQ.eval_transform
#         else:
#             self.transform = transform
#         self.env = env
#         self.bias_amount=bias_amount
#         self.n_classes = 2
#         self.return_index = return_index
#         self.target_name = target_name
#         self.confounder_names = confounder_names
#         self.n_confounders = 1
#         self.n_groups = 4

#         self.bias_folder_dict = {
#             0.995: "0.5pct"
#         }

#         if not os.path.isdir(os.path.join(self.root, "bffhq")):
#             self.__download_dataset()
#         else: self.root = os.path.join(self.root, "bffhq")

#         if self.env == "train":
#             self.filename_array, self.y_array, self.group_array = self.load_train_samples()
#             if external_bias_labels:
#                 print("Loading external bias labels for the training set...")
#                 self.old_garray = self.group_array.copy()
#                 self.group_array = pd.read_csv("bffhq_metadata_aug.csv")["ddb"].to_numpy()
#                 assert len(self.old_garray) == len(self.group_array)
#                 print(np.sum(self.old_garray != self.group_array)/len(self.group_array))

#         if self.env == "val":
#             self.filename_array, self.y_array, self.group_array = self.load_val_samples()

#         if self.env == "test":
#             self.filename_array, self.y_array, self.group_array = self.load_test_samples()

#     def __download_dataset(self) -> None:
#         os.makedirs(self.root, exist_ok=True)
#         output_path = os.path.join(self.root, "bffhq.zip")
#         print(f"=> Downloading {BFFHQ.DATASET_NAME} for {BFFHQ.DOWNLOAD_URL}")

#         try:
#             gdown.download(id="1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp", output=output_path)
#         except:
#             raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
#         print(f"=> Extracting bffhq.zip to directory {self.root}")
#         try:
#             with zipfile.ZipFile(output_path, mode="r") as unzipper:
#                 unzipper.extractall(self.root)
#         except:
#             raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

#         self.root = os.path.join(self.root, "bffhq")
#         os.remove(output_path)

#     def __len__(self):
#         return len(self.filename_array)
    
#     def __getitem__(self, index):
#         file_path = self.filename_array[index]
#         class_label = self.y_array[index]
#         bias_label = self.group_array[index]

#         image = self.transform(Image.open(file_path))
        
#         return image, (class_label, bias_label), index        

#     def load_train_samples(self):
#         samples_path:   List[str] = []
#         class_labels:   List[int] = []
#         bias_labels:    List[int] = []

#         bias_folder = self.bias_folder_dict[self.bias_amount]
        
#         for class_folder in sorted(os.listdir(os.path.join(self.root, bias_folder, "align"))):
#             for filename in sorted(os.listdir(os.path.join(self.root, bias_folder, "align", class_folder))):
#                 samples_path.append(os.path.join(self.root, bias_folder, "align", class_folder, filename))
#                 class_labels.append(self.assign_class_label(filename))
#                 bias_labels.append(self.assign_bias_label(filename))

#         for class_folder in sorted(os.listdir(os.path.join(self.root, bias_folder, "conflict"))):
#             for filename in sorted(os.listdir(os.path.join(self.root, bias_folder, "conflict", class_folder))):
#                 samples_path.append(os.path.join(self.root, bias_folder, "conflict", class_folder, filename))
#                 class_labels.append(self.assign_class_label(filename))
#                 bias_labels.append(self.assign_bias_label(filename))     

#         return (
#             np.array(samples_path),
#             np.array(class_labels),
#             np.array(bias_labels)
#         )
    
#     def load_val_samples(self):
#         samples_path:   List[str] = []
#         class_labels:   List[int] = []
#         bias_labels:    List[int] = []

#         bias_folder = self.bias_folder_dict[self.bias_amount]

#         for filename in sorted(os.listdir(os.path.join(self.root, "valid"))):
#             samples_path.append(os.path.join(self.root, "valid", filename))
#             class_labels.append(self.assign_class_label(filename))
#             bias_labels.append(self.assign_bias_label(filename))

#         return (
#             np.array(samples_path),
#             np.array(class_labels),
#             np.array(bias_labels)
#         )
    
#     def load_test_samples(self):
#         samples_path:   List[str] = []
#         class_labels:   List[int] = []
#         bias_labels:    List[int] = []

#         for filename in sorted(os.listdir(os.path.join(self.root, "test"))):
#                 samples_path.append(os.path.join(self.root, "test", filename))
#                 class_labels.append(self.assign_class_label(filename))
#                 bias_labels.append(self.assign_bias_label(filename))

#         return (
#             np.array(samples_path),
#             np.array(class_labels),
#             np.array(bias_labels)
#         )
    
#     def assign_bias_label(self, filename: str) -> int:
#         no_extension = filename.split(".")[0]
#         _, _, z = no_extension.split("_")
#         return int(z)
    
#     def assign_class_label(self, filename: str) -> int:
#         no_extension = filename.split(".")[0]
#         _, y, _ = no_extension.split("_")
#         return int(y)
    
#     def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
#         labels: torch.Tensor = torch.zeros(len(self))
#         for i in range(len(self)):
#             labels[i] = self[i][1][0]

#         _, pop_counts = labels.unique(return_counts=True)

#         if return_labels:
#             return pop_counts.long(), labels.long()

#         return pop_counts
    
#     def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
#         for i in range(len(self)):
#             yield self[i][1][1]

#     def get_class_labels(self) -> Generator[None, None, torch.Tensor]:
#         for i in range(len(self)):
#             yield self[i][1][0]
    

#     def get_splits(self, splits, train_frac=1.0):
#         subsets = {}
#         for split in splits:
#             assert split in ("train", "val", "test"), f"{split} is not a valid split"
#             split_set = BFFHQ(env=split, return_index=True)
#             subsets[split] = split_set

#         return subsets
    
#     def group_str(self, group_idx):
#         y = group_idx // (self.n_groups / self.n_classes)
#         c = group_idx % (self.n_groups // self.n_classes)

#         group_name = str(y)
#         bin_str = format(int(c), f"{self.n_confounders}")[::-1]
#         return group_name
    
#     def get_group_array(self):
#         return self.group_array
    
#     def get_label_array(self):
#         return self.y_array
    
    
#     def __repr__(self) -> str:
#         return f"BFFHQ(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.n_classes})"
    
# if __name__ == "__main__":
#     d = BFFHQ()
#     print("a")


#!/usr/bin/env python3

import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator, Union
import gdown
import requests
import zipfile
from tqdm import tqdm
import pandas as pd

class BFFHQ(Dataset):
    DOWNLOAD_URL = "https://drive.google.com/file/d/1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp/view?usp=sharing"
    DATASET_NAME = "bffhq"    

    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(
            self, 
            root_dir="./data", 
            env="train", 
            bias_amount=99.5,
            target_name="age",
            confounder_names="gender",
            return_index = False,
            transform=None,
            external_bias_labels: bool = True,
            **kwargs
        ):
        self.root = "./data"
        if transform is None:
            self.transform = BFFHQ.train_transform if env == "train" else BFFHQ.eval_transform
        else:
            self.transform = transform
        self.env = env
        self.bias_amount=bias_amount
        self.n_classes = 2
        self.return_index = return_index
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.n_confounders = 1
        self.n_groups = 4

        self.bias_folder_dict = {
            99.5: "0.5pct"
        }

        if not os.path.isdir(os.path.join(self.root, "bffhq")):
            self.__download_dataset()
        else: self.root = os.path.join(self.root, "bffhq")

        if self.env == "train":
            self.filename_array, self.y_array, self.confounder_array = self.load_train_samples()
            if external_bias_labels:
                print("Loading external bias labels for the training set...")
                self.old_garray = self.confounder_array.copy()
                self.confounder_array = pd.read_csv(os.path.join("outputs", "bffhq_metadata_aug.csv"), header="infer")["ddb"].to_numpy()
                assert len(self.old_garray) == len(self.confounder_array)
                self.old_garray = None

        if self.env == "val":
            self.filename_array, self.y_array, self.confounder_array = self.load_val_samples()

        if self.env == "test":
            self.filename_array, self.y_array, self.confounder_array = self.load_test_samples()

        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

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

        self.root = os.path.join(self.root, "bffhq")
        os.remove(output_path)

    def __len__(self):
        return len(self.filename_array)
    
    def __getitem__(self, index):
        file_path = self.filename_array[index]
        class_label = self.y_array[index]
        bias_label = self.confounder_array[index]

        image = self.transform(Image.open(file_path))
        
        return image, (class_label, bias_label), index        

    def load_train_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]
        
        for class_folder in sorted(os.listdir(os.path.join(self.root, bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.root, bias_folder, "align", class_folder))):
                samples_path.append(os.path.join(self.root, bias_folder, "align", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        for class_folder in sorted(os.listdir(os.path.join(self.root, bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.root, bias_folder, "conflict", class_folder))):
                samples_path.append(os.path.join(self.root, bias_folder, "conflict", class_folder, filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))     

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_val_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]

        for filename in sorted(os.listdir(os.path.join(self.root, "valid"))):
            samples_path.append(os.path.join(self.root, "valid", filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def load_test_samples(self):
        samples_path:   List[str] = []
        class_labels:   List[int] = []
        bias_labels:    List[int] = []

        for filename in sorted(os.listdir(os.path.join(self.root, "test"))):
                samples_path.append(os.path.join(self.root, "test", filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels)
        )
    
    def assign_bias_label(self, filename: str) -> int:
        no_extension = filename.split(".")[0]
        _, _, z = no_extension.split("_")
        return int(z)
    
    def assign_class_label(self, filename: str) -> int:
        no_extension = filename.split(".")[0]
        _, y, _ = no_extension.split("_")
        return int(y)
    
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
            split_set = BFFHQ(env=split, return_index=True)
            subsets[split] = split_set

        return subsets
    
    def group_str(self, group_idx):
        # Calculate the class (y) and confounder (c) for the group index
        y = group_idx // (self.n_groups // self.n_classes)  # Class label
        c = group_idx % (self.n_groups // self.n_classes)   # Confounder

        # Convert confounder to binary string format with leading zeros (based on n_confounders)
        bin_str = format(c, f"0{self.n_confounders}b")

        # Combine class and confounder into the group name
        group_name = f"Class: {int(y)}, Confounder: {c}"
        
        return group_name
    
    def get_group_array(self):
        return self.group_array
    
    def get_label_array(self):
        return self.y_array
    
    
    def __repr__(self) -> str:
        return f"BFFHQ(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.n_classes})"
    
if __name__ == "__main__":
    d = BFFHQ()
    print("a")
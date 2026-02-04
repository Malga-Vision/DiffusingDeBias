# import gdown
# import tarfile
# import os

# from glob import glob
# from torch.utils.data.dataset import Dataset
# from PIL import Image
# import gzip



# class CorruptedCIFAR10(Dataset):
#     def __init__(self, root="./data", split="train", percent="5pct", transform=None, image_path_list=None):
#         super().__init__()
        
#         if not os.path.isdir(os.path.join(root, 'cifar10c')):
#             self.download_dataset(root)
#         root = os.path.join(root, 'cifar10c', percent)

#         self.transform = transform
#         self.root = root
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         if split == 'train':
#             self.align = glob(os.path.join(root, 'align',"*","*"))
#             self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
#             self.data = self.align + self.conflict

#         elif split == 'valid':
#             self.data = glob(os.path.join(root,split,"*", "*"))

#         elif split == 'test':
#             self.data = glob(os.path.join(root, '../test',"*","*"))

#     def download_dataset(self, path):
#         url = "https://drive.google.com/file/d/1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-/view?usp=sharing"
#         output = os.path.join(path, 'cifar10c.tar.gz')
#         print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
#         gdown.download(id="1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-", output=output, quiet=False)

#         print('=> Extracting dataset..')
#         with tarfile.open(os.path.join(path, 'cifar10c.tar.gz'), 'r:gz') as tar:
#             tar.extractall(path=path)
    
#         os.remove(output)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         label, bias = int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])
#         # bias = int(bias != label)
#         image = Image.open(self.data[index]).convert('RGB')
#         # image = Image.open(self.data[index])
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, (label, bias), index

# if __name__ == '__main__':
#     pass
from typing import Generator, Tuple, Union
import gdown
import tarfile
import os

from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch


class CorruptedCIFAR10(Dataset):

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
            bias_amount = 95.0, 
            target_name = "object", 
            confounder_names="corruption", 
            return_index = False, 
            transform=None, 
            image_path_list=None, 
            external_bias_labels: bool = True,
            **kwargs
        ):
        super().__init__()
        
        if not os.path.isdir(os.path.join(root_dir, 'cifar10c')):
            self.download_dataset(root_dir)
        
        converter = {
            95.0: "5pct",
            98.0: "2pct",
            99.5: "0.5pct"
        }

        self.bias_amount = bias_amount
        root_dir = os.path.join(root_dir, 'cifar10c', converter[bias_amount])
        print(root_dir)

        if transform is None:
            self.transform = CorruptedCIFAR10.train_transform if env == "train" else CorruptedCIFAR10.eval_transform
        else: 
            self.transform = transform
        
        self.env = env
        self.root_dir = root_dir
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.confounder_array = None
        self.n_classes = 10
        self.n_groups = 20
        self.n_confounders = 1
        self.return_index = return_index
            
        
        if self.env == 'train':
            self.align = glob(os.path.join(root_dir, 'align',"*","*"))
            self.conflict = glob(os.path.join(root_dir, 'conflict',"*","*"))
            self.filename_array = self.align + self.conflict
        
        elif self.env == 'val':
            self.filename_array = glob(os.path.join(root_dir, "valid", "*", "*"))

        elif self.env == 'test':
            self.filename_array = glob(os.path.join(root_dir, '../test',"*","*"))

        
        self.y_array = torch.as_tensor([
            int(self.filename_array[index].split('_')[-2]) for index in range(len(self.filename_array))
        ])
        
        self.confounder_array = torch.as_tensor([
            int(self.filename_array[index].split('_')[-1].split('.')[0]) for index in range(len(self.filename_array))
        ]).long()


        if external_bias_labels:
            if self.env == "train":            
                print("Loading external bias labels!")
                self.confounder_array = pd.read_csv(os.path.join("outputs", f"cifar10c_{bias_amount}_metadata_aug.csv"))["ddb"].to_numpy()
                print("Loaded ", f"cifar10c_{bias_amount}_metadata_aug.csv")
            else:
                print("External bias labels are supported only for the training set, skipping...")

        self.group_array = ((self.y_array * 2) + self.confounder_array).long()
        
    def download_dataset(self, path):
        url = "https://drive.google.com/file/d/1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-/view?usp=sharing"
        output = os.path.join(path, 'cifar10c.tar.gz')
        print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
        gdown.download(id="1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-", output=output, quiet=False)

        print('=> Extracting dataset..')
        with tarfile.open(os.path.join(path, 'cifar10c.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=path)
    
        os.remove(output)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, index):
        file_path = self.filename_array[index]
        class_label = self.y_array[index]
        bias_label = self.group_array[index]#int(self.confounder_array[index] == class_label)

        image = self.transform(Image.open(file_path).convert("RGB"))
        
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
            yield self[i][1][1]

    def get_class_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][1][0]

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), f"{split} is not a valid split"
            split_set = CorruptedCIFAR10(env=split, return_index=True, bias_amount=self.bias_amount)
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

if __name__ == '__main__':
    pass

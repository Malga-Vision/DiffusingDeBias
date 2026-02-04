import gdown
import tarfile
import os
import numpy as np
from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2



class CorruptedCIFAR10(Dataset):
    def __init__(self, root, split, percent, transform=None, image_path_list=None, class_label: int = None):
        super().__init__()
        
        if not os.path.isdir(os.path.join(root, 'cifar10c')):
            self.download_dataset(root)
        root = os.path.join(root, 'cifar10c', percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.num_classes = 10 if class_label is None else 1

        if split == 'train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))

            if class_label is not None and class_label < self.num_classes:
                self.align = [x for x in self.align if int(x.split('_')[-2]) == class_label]
                self.conflict = [x for x in self.conflict if int(x.split('_')[-2]) == class_label]
                
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

            if class_label is not None and class_label < self.num_classes:
                self.data = [x for x in self.data if int(x.split('_')[-2]) == class_label]
                
        elif split == 'test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

            if class_label is not None and class_label < self.num_classes:
                self.data = [x for x in self.data if int(x.split('_')[-2]) == class_label]

    def download_dataset(self, path):
        url = "https://drive.google.com/uc?id=1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-"

        output = os.path.join(path, 'cifar10c.tar.gz')
        print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
        gdown.download(url, output, quiet=False, fuzzy=True)

        print('Extracting dataset...')
        with tarfile.open(output, 'r:gz') as tar:
            tar.extractall(path=path)
        
        os.remove(output)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data[index]
        class_label, bias_label = int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])


        image=np.array(Image.open(file_path))
        if self.transform is not None:
            image = self.transform(image=image)["image"] 

        data_dict = {
            'name': file_path, 
            'image': image, 
            'class_label': class_label, 
            'bias_label': bias_label
        }  
        
        return data_dict

if __name__ == '__main__':
    dataset = CorruptedCIFAR10(f'/home/{os.environ.get("USER")}/temp', 'train', "0.5pct")
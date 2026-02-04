import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Generator, Union, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class ImageNet9(Dataset):

    def __init__(self, root="./data/imagenet9", transform=None, return_index=False, class_label=None):
        self.root = root
        self.transform = transform
        self.return_index = return_index
        self.num_classes = 9 if class_label is None else 1

        self.samples, self.class_labels = self._load_samples(class_label)
        print(f"Loaded {len(self.samples)} samples")
        print(f"Loaded {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.samples[idx]
        assert os.path.isfile(file_path), f"File not found: {file_path}"

        class_label = self.class_labels[idx]

        with Image.open(file_path) as img:
            np_image = np.array(img.convert('RGB'))
    
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=np_image)
            image = transformed["image"]
        else:
            raise ValueError("Transforms must be provided")
        
        data_dict = {
            'name': os.path.basename(file_path),
            'image': image,
            'class_label': class_label,
            'bias_label': -1
        }

        if self.return_index:
            data_dict['index'] = idx

        return data_dict

    def _load_samples(self, class_label: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = []
        class_labels = []

        # Load CSV file
        csv_path = os.path.join(self.root, "imagenet_9.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # Extract filename from path and check extension
            filename = os.path.basename(row['path'])
            if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
                print(f"Skipping non-image file {filename}")
                continue
            
            # Construct full image path
            file_path = os.path.join(self.root, filename)
            assert os.path.isfile(file_path), f"File not found: {file_path}"                
            
            class_idx = row['target']

            # Apply class filter if specified
            if class_label is not None and class_idx != class_label:
                continue
            
            samples.append(file_path)
            class_labels.append(class_idx)
        
        return np.array(samples), np.array(class_labels)

    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        labels = torch.tensor(self.class_labels)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        return (counts, unique_labels) if return_labels else counts

    def __repr__(self) -> str:
        return f"ImageNet9(num_samples={len(self)}, num_classes={self.num_classes})"


if __name__ == "__main__":
    # Define transformations
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Initialize dataset
    dataset = ImageNet9(
        root="/home/roberto/Datasets/Bias/imagenet9-rebias/imagenet9",
        transform=transform
    )

    print(f"Total dataset size: {len(dataset)}")
    print(f"Class populations: {dataset.perclass_populations(return_labels=True)}")

    # Create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Print sample shapes
    sample = next(iter(loader))
    print("\nSample shapes:")
    print(f"Images: {sample['image'].shape}")
    print(f"Class labels: {sample['class_label'].shape}")
    print(f"Bias labels: {sample['bias_label'].shape}")

    import matplotlib.pyplot as plt  # Add this import at the top

    def visualize_samples(dataset, samples_per_class=5, filename="class_samples.png"):
        """
        Visualizes samples_per_class images for each target class and saves to disk.
        
        Args:
            dataset (ImageNet9): The dataset to visualize
            samples_per_class (int): Number of samples to display per class
            filename (str): Output path to save the visualization
        """
        import numpy as np
        import os

        # Create output directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        class_labels = dataset.class_labels
        unique_classes = np.unique(class_labels)
        num_classes = len(unique_classes)
        
        # Create figure and axes
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(2*samples_per_class, 2*num_classes),
                                squeeze=False)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        
        # Create dictionary mapping classes to their indices
        class_to_indices = {cls: np.where(class_labels == cls)[0] for cls in unique_classes}
        
        for i, cls in enumerate(unique_classes):
            indices = class_to_indices[cls][:samples_per_class]
            for j in range(samples_per_class):
                ax = axes[i, j]
                if j < len(indices):
                    data = dataset[indices[j]]
                    image = data['image']
                    
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = image.numpy().transpose(1, 2, 0)
                    image_np = std * image_np + mean
                    image_np = np.clip(image_np, 0, 1)
                    
                    ax.imshow(image_np)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel(f'Class {cls}', rotation=0, ha='right', va='center')
                else:
                    ax.axis('off')

        # Save and close the figure
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved visualization to {filename}")
        
    visualize_samples(dataset, filename="./class_samples.png")

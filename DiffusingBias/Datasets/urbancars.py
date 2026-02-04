import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UrbanCars(Dataset):
    """
    Dataset class for UrbanCars which contains images of urban and country cars
    with different backgrounds and co-occurring objects, designed to study
    shortcut learning and bias in computer vision models.
    """
    base_folder = "urbancars_images"

    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        group_label="both",
        return_group_index=False,
        return_domain_label=False,
        return_dist_shift=False,
        class_label: int = None,
    ):
        """
        Initialize the UrbanCars dataset.
        
        Args:
            root (str): Root directory of dataset
            split (str): "train", "val", or "test"
            transform: Image transforms (e.g., albumentations transforms)
            group_label (str): How to group the data ("bg", "co_occur_obj", or "both")
            return_group_index (bool): Whether to return group index
            return_domain_label (bool): Whether to return domain label
            return_dist_shift (bool): Whether to return distribution shift
            class_label (int, optional): If specified, only load this class
        """
        # Set ratios based on split
        if split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif split in ["val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError(f"Split {split} is not implemented")
        
        self.bg_ratio = bg_ratio
        self.co_occur_obj_ratio = co_occur_obj_ratio
        self.split = split
        self.num_classes = len(self.obj_name_list) if class_label is None else 1
        self.class_label = class_label

        assert os.path.exists(os.path.join(root, self.base_folder))
        assert group_label in ["bg", "co_occur_obj", "both"]
        self.group_label = group_label
        self.transform = transform
        self.return_group_index = return_group_index
        self.return_domain_label = return_domain_label
        self.return_dist_shift = return_dist_shift

        # Construct path to the split directory
        ratio_combination_folder_name = f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        self.img_root = os.path.join(root, self.base_folder, ratio_combination_folder_name, split)
        
        # Collect samples and labels
        self._load_samples()
        
        # Set up group label information
        if group_label == "bg":
            num_shortcut_category = len(self.bg_name_list)
            shortcut_label = self.bg_labels
        elif group_label == "co_occur_obj":
            num_shortcut_category = len(self.co_occur_obj_name_list)
            shortcut_label = self.co_occur_obj_labels
        elif group_label == "both":
            num_shortcut_category = len(self.bg_name_list) * len(self.co_occur_obj_name_list)
            shortcut_label = self.bg_labels * len(self.co_occur_obj_name_list) + self.co_occur_obj_labels
        else:
            raise NotImplementedError

        self.domain_label = shortcut_label
        self.set_num_group_and_group_array(num_shortcut_category, shortcut_label)
        
        print(f"Loaded {len(self.img_paths)} samples from {self.img_root}")

    def _load_samples(self):
        """Load all samples and labels"""
        self.img_paths = []
        self.obj_labels = []
        self.bg_labels = []
        self.co_occur_obj_labels = []
        
        for obj_id, obj_name in enumerate(self.obj_name_list):
            # Skip if we only want a specific class
            if self.class_label is not None and obj_id != self.class_label:
                continue
                
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(self.co_occur_obj_name_list):
                    dir_name = f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    dir_path = os.path.join(self.img_root, dir_name)
                    
                    if not os.path.exists(dir_path):
                        print(f"Warning: Directory {dir_path} does not exist")
                        continue
                    
                    # Get all jpg files in the directory
                    for file_name in os.listdir(dir_path):
                        if file_name.endswith('.jpg') and not file_name.endswith('_mask.jpg') and not file_name.endswith('_bg_only.jpg') and not file_name.endswith('_co_occur_obj_mask.jpg'):
                            img_path = os.path.join(dir_path, file_name)
                            self.img_paths.append(img_path)
                            self.obj_labels.append(obj_id)
                            self.bg_labels.append(bg_id)
                            self.co_occur_obj_labels.append(co_occur_obj_id)
        
        # Convert to tensors
        self.obj_labels = torch.tensor(self.obj_labels, dtype=torch.long)
        self.bg_labels = torch.tensor(self.bg_labels, dtype=torch.long)
        self.co_occur_obj_labels = torch.tensor(self.co_occur_obj_labels, dtype=torch.long)

    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        """Set up group information"""
        self.num_group = len(self.obj_name_list) * num_shortcut_category
        self.group_array = self.obj_labels * num_shortcut_category + shortcut_label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        obj_label = self.obj_labels[index].item()
        bg_label = self.bg_labels[index].item()
        co_occur_obj_label = self.co_occur_obj_labels[index].item()
        
        # Set bias label based on group label
        if self.group_label == "bg":
            bias_label = 1 if obj_label == bg_label else -1
        elif self.group_label == "co_occur_obj":
            bias_label = 1 if obj_label == co_occur_obj_label else -1
        elif self.group_label == "both":
            bias_label = 1 if (obj_label == bg_label and obj_label == co_occur_obj_label) else -1
        
        # Load and transform the image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            transformed = self.transform(image=np.array(img))
            img = transformed["image"]
        
        # Prepare the data dictionary
        data_dict = {
            'name': img_path,
            'image': img,
            'class_label': obj_label,
            'bias_label': bias_label,
            'bg_label': bg_label,
            'co_occur_obj_label': co_occur_obj_label
        }

        if self.return_group_index:
            data_dict["group_index"] = self.group_array[index].item()

        if self.return_domain_label:
            data_dict["domain_label"] = self.domain_label[index].item()

        if self.return_dist_shift:
            data_dict["dist_shift"] = 0
        
        return data_dict

    def _get_subsample_group_indices(self, subsample_which_shortcut):
        """Get indices for balanced subsampling"""
        bg_ratio = self.bg_ratio
        co_occur_obj_ratio = self.co_occur_obj_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)
        if subsample_which_shortcut == "bg":
            min_size = int(min(1 - bg_ratio, bg_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "co_occur_obj":
            min_size = int(min(1 - co_occur_obj_ratio, co_occur_obj_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "both":
            min_bg_ratio = min(1 - bg_ratio, bg_ratio)
            min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
            min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)
        else:
            raise NotImplementedError

        assert min_size > 1

        indices = []
        obj_bg_co_occur_obj_tensor = torch.stack([self.obj_labels, self.bg_labels, self.co_occur_obj_labels], dim=1)

        if subsample_which_shortcut == "bg":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = obj_bg_co_occur_obj_tensor[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = obj_bg_co_occur_obj_tensor[:, 1] == idx_bg
                    mask = obj_mask & bg_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    if isinstance(subgroup_indices, int):  # Handle case when only one index is found
                        subgroup_indices = [subgroup_indices]

                    indices_tensor = torch.tensor(subgroup_indices)
                    perm = torch.randperm(len(indices_tensor))
                    sampled_indices = indices_tensor[perm[:min_size]].tolist()
                    indices += sampled_indices
                    
        elif subsample_which_shortcut == "co_occur_obj":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = obj_bg_co_occur_obj_tensor[:, 0] == idx_obj
                for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                    co_occur_obj_mask = obj_bg_co_occur_obj_tensor[:, 2] == idx_co_occur_obj
                    mask = obj_mask & co_occur_obj_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    if isinstance(subgroup_indices, int):
                        subgroup_indices = [subgroup_indices]

                    indices_tensor = torch.tensor(subgroup_indices)
                    perm = torch.randperm(len(indices_tensor))
                    sampled_indices = indices_tensor[perm[:min_size]].tolist()
                    indices += sampled_indices
                    
        elif subsample_which_shortcut == "both":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = obj_bg_co_occur_obj_tensor[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = obj_bg_co_occur_obj_tensor[:, 1] == idx_bg
                    for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                        co_occur_obj_mask = obj_bg_co_occur_obj_tensor[:, 2] == idx_co_occur_obj
                        mask = obj_mask & bg_mask & co_occur_obj_mask
                        subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                        if isinstance(subgroup_indices, int):
                            subgroup_indices = [subgroup_indices]
                        if not subgroup_indices:  # Skip if empty
                            continue

                        indices_tensor = torch.tensor(subgroup_indices)
                        perm = torch.randperm(len(indices_tensor))  
                        if min_size <= len(indices_tensor):
                            sampled_indices = indices_tensor[perm[:min_size]].tolist()
                        else:
                            sampled_indices = indices_tensor.tolist()
                        indices += sampled_indices
        
        return indices

    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the population counts per class and per bias group."""
        # Class population
        unique_class_labels, class_counts = torch.unique(self.obj_labels, return_counts=True)

        # Bias population
        # Determine bias labels for all samples based on self.group_label
        if self.group_label == "bg":
            # Biased if object class matches background class
            bias_labels_tensor = torch.where(self.obj_labels == self.bg_labels, 1, -1)
        elif self.group_label == "co_occur_obj":
            # Biased if object class matches co-occurring object class
            bias_labels_tensor = torch.where(self.obj_labels == self.co_occur_obj_labels, 1, -1)
        elif self.group_label == "both":
            # Biased if object class matches both background and co-occurring object class
            condition = (self.obj_labels == self.bg_labels) & (self.obj_labels == self.co_occur_obj_labels)
            bias_labels_tensor = torch.where(condition, 1, -1)
        else:
            # This case should be prevented by the __init__ assertion
            raise ValueError(f"Invalid group_label specified: {self.group_label}") 

        unique_bias_values, bias_counts = torch.unique(bias_labels_tensor, return_counts=True)

        if return_labels:
            # Return class counts, unique class labels, bias counts, unique bias labels
            return class_counts, unique_class_labels, bias_counts, unique_bias_values
        else:
            # Return class counts, bias counts
            return class_counts, bias_counts

    def get_bias_labels(self) -> Generator[int, None, None]:
        """Generate bias labels for all samples based on the current group_label setting."""
        for i in range(len(self.img_paths)): # Iterate through all samples
            obj_label = self.obj_labels[i].item()
            bg_label = self.bg_labels[i].item()
            co_occur_obj_label = self.co_occur_obj_labels[i].item()
            
            current_bias_label = 0 # Default or placeholder, should be overwritten
            if self.group_label == "bg":
                current_bias_label = 1 if obj_label == bg_label else -1
            elif self.group_label == "co_occur_obj":
                current_bias_label = 1 if obj_label == co_occur_obj_label else -1
            elif self.group_label == "both":
                current_bias_label = 1 if (obj_label == bg_label and obj_label == co_occur_obj_label) else -1
            else:
                # This path should not be reached if __init__ validates group_label
                raise ValueError(f"Invalid group_label: {self.group_label}")

            yield current_bias_label

    def get_sampling_weights(self):
        """Return sampling weights to balance the dataset"""
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights
            
    def __repr__(self) -> str:
        return f"UrbanCars(split={self.split}, bg_ratio={self.bg_ratio}, co_occur_obj_ratio={self.co_occur_obj_ratio}, num_samples={len(self)})"


# Example usage
if __name__ == "__main__":
    # Set seed
    seed = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define transformations using albumentations
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(normalization="min_max_per_channel"),
        ToTensorV2(),
    ])
    
    # Initialize dataset
    dataset = UrbanCars(
        root="/home/roberto/Datasets/Bias/urbancars",
        split="train",
        group_label="both",
        transform=transform,
        return_group_index=False,
        return_domain_label=False,
        return_dist_shift=False,
        class_label=None
    )
    
    print(f"Dataset: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Class label: {sample['class_label']}")
    print(f"Bias label: {sample['bias_label']}")
    
    # Get population counts
    class_pop_counts, unique_classes, bias_pop_counts, unique_biases = dataset.perclass_populations(return_labels=True)
    print(f"Per class populations: Counts: {class_pop_counts}, Labels: {unique_classes}")
    print(f"Per bias populations: Counts: {bias_pop_counts}, Labels: {unique_biases} (1: biased, -1: unbiased for '{dataset.group_label}' grouping)")
    
    # Create balanced subset
    balanced_indices = dataset._get_subsample_group_indices("both")
    print(f"Number of balanced samples: {len(balanced_indices)}")


    # Visualize some samples
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(25, 25))
    
    # Determine the number of samples to visualize
    # Ensure we don't try to access more samples than available in balanced_indices or more than the grid size
    num_samples_to_visualize = min(len(balanced_indices), len(axes.flatten()))

    for i in range(num_samples_to_visualize):
        ax = axes.flatten()[i]
        sample_idx = balanced_indices[i] # Use index from the balanced list
        sample = dataset[sample_idx]    # Get sample using the balanced index
        ax.imshow(sample['image'].permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"{os.path.basename(sample['name'])}\nClass: {sample['class_label']}, Bias: {sample['bias_label']}")
        ax.axis('off') # Turn off axis numbers and ticks for a cleaner look
    plt.tight_layout()
    plt.savefig("urbancars_samples.png")
    plt.close()

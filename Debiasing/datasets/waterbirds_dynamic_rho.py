import os
import tarfile
from typing import Tuple, Union, Dict, List, Generator, Literal

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from tabulate import tabulate

Strategy = Literal["downsample", "oversample", "borrow_from_val"]

class WaterbirdsDynamicRho(Dataset):
    """
    Waterbirds dataset with adjustable training set bias via different strategies.

    This class can be used in two ways:
    1. As a standard Dataset loader for a given split (train/val/test).
    2. To generate new, complete metadata files with a modified training set
       using the `create_and_save_metadata` static method.

    Args:
        env (str): The dataset split, one of ["train", "val", "test"].
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform to apply to the images.
        bias_amount (float): The desired correlation for the training set (e.g., 0.95, 0.70).
                             Only used when creating a new training set.
        strategy (Strategy): The resampling strategy for the training set.
        metadata_path (str): Path to the metadata CSV file to use.
        random_seed (int): The random seed for resampling the dataset.
    """
    DOWNLOAD_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    DATASET_NAME = "waterbirds"

    def __init__(self,
                 env: str,
                 root: str = "./data",
                 transform=None,
                 bias_amount: float = 0.95,
                 strategy: Strategy = "downsample",
                 metadata_path: str = None,
                 random_seed: int = 42):

        self.root: str = os.path.join(root, WaterbirdsDynamicRho.DATASET_NAME)
        self.env: str = env
        self.transform = transform
        self.num_classes = 2
        self.n_groups = 4
        self.env_to_split = {"train": 0, "val": 1, "test": 2}

        self.data_dir = os.path.join(self.root, "waterbird_complete95_forest2water2")
        if not os.path.exists(self.data_dir):
            self.__download_dataset()

        # If no specific metadata path is given, use the default one.
        if metadata_path is None:
            metadata_path = os.path.join(self.data_dir, "metadata.csv")

        if not os.path.exists(metadata_path):
             raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"You may need to generate it first using `create_and_save_metadata`."
            )

        # Load the specified metadata and filter for the environment (train/val/test)
        metadata_df = pd.read_csv(metadata_path)
        self.metadata = metadata_df.query(f"split == {self.env_to_split[self.env]}")

        self.samples = []
        for _, row in self.metadata.iterrows():
            self.samples.append({
                "image_path":  os.path.join(self.data_dir, row["img_filename"]),
                "class_label": int(row["y"]),
                "bias_label":  int(row["place"]), # here it is not used other than evaluation purposes
            })

    @staticmethod
    def create_and_save_metadata(
        output_path: str,
        strategy: Strategy,
        bias_amount: float,
        root: str = "./data",
        random_seed: int = 0
    ):
        """
        Generates a new metadata file with a resampled training split and saves it.
        This is a static method, so it can be called directly from the class.
        """
        print(f"Generating new metadata with bias={bias_amount}, strategy='{strategy}'...")
        
        # Instantiate the class to ensure data is downloaded
        base_data_dir = os.path.join(root, WaterbirdsDynamicRho.DATASET_NAME, "waterbird_complete95_forest2water2")
        if not os.path.exists(base_data_dir):
            WaterbirdsDynamicRho(env="test", root=root) # Triggers download if needed
            
        original_metadata_path = os.path.join(base_data_dir, "metadata.csv")
        original_df = pd.read_csv(original_metadata_path)

        # 1. Define the pool of data to sample from
        train_df = original_df.query("split == 0")
        sampling_pool = train_df
        if strategy == "borrow_from_val":
            print("=> Strategy: Borrowing from validation set.")
            val_df = original_df.query("split == 1")
            sampling_pool = pd.concat([train_df, val_df])

        # 2. Resample the pool to create the new training data
        new_train_df = WaterbirdsDynamicRho._resample_pool(
            pool=sampling_pool,
            original_train_df=train_df,
            target_bias=bias_amount,
            strategy=strategy,
            seed=random_seed
        )
        new_train_df['split'] = 0

        # 3. Get the other splits
        test_df = original_df.query("split == 2").copy()
        # If we borrowed from val, the new val split is empty
        val_df = pd.DataFrame(columns=original_df.columns) if strategy == "borrow_from_val" else original_df.query("split == 1").copy()

        # 4. Combine and save
        final_metadata = pd.concat([new_train_df, val_df, test_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_metadata.to_csv(output_path, index=False)
        print(f"Successfully saved new metadata to: {output_path}")
        print("\nNew split distribution:")
        print(final_metadata['split'].value_counts().sort_index())


    @staticmethod
    def _resample_pool(pool: pd.DataFrame, original_train_df: pd.DataFrame, target_bias: float, strategy: Strategy, seed: int) -> pd.DataFrame:
        """Static helper method to resample a dataframe to a target bias."""
        print(f"=> Adjusting training set bias to {target_bias:.2f} using '{strategy}' strategy.")
        pool_groups = {(y, p): pool[(pool['y'] == y) & (pool['place'] == p)] for y in [0, 1] for p in [0, 1]}

        if strategy == "downsample":
            n_minority_lb = len(original_train_df[(original_train_df['y'] == 0) & (original_train_df['place'] == 1)])
            n_minority_wb = len(original_train_df[(original_train_df['y'] == 1) & (original_train_df['place'] == 0)])
            target_counts = {
                (0, 0): int(n_minority_lb * target_bias / (1.0 - target_bias)),
                (0, 1): n_minority_lb,
                (1, 0): n_minority_wb,
                (1, 1): int(n_minority_wb * target_bias / (1.0 - target_bias)),
            }
        else:  # "oversample" or "borrow_from_val"
            class_counts = original_train_df['y'].value_counts()
            target_counts = {
                (0, 1): int(class_counts[0] * (1 - target_bias)),
                (0, 0): int(class_counts[0] * target_bias),
                (1, 0): int(class_counts[1] * (1 - target_bias)),
                (1, 1): int(class_counts[1] * target_bias),
            }
        
        sampled_groups = []
        for (y, p), n_target in target_counts.items():
            group_pool = pool_groups[(y, p)]
            replace = n_target > len(group_pool)
            if replace and strategy != "downsample":
                print(f"   - Oversampling group (y={y}, p={p}) from {len(group_pool)} to {n_target}")
            sampled_groups.append(group_pool.sample(n=n_target, random_state=seed, replace=replace))

        final_df = pd.concat(sampled_groups)
        return final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "waterbirds.tar.gz")
        print(f"=> Downloading {WaterbirdsDynamicRho.DATASET_NAME} from {WaterbirdsDynamicRho.DOWNLOAD_URL}")
        try:
            response = requests.get(WaterbirdsDynamicRho.DOWNLOAD_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(output_path, "wb") as f, tqdm(desc=output_path, total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except requests.RequestException as e:
            raise RuntimeError(f"Unable to complete dataset download: {e}")
        print(f"=> Extracting {output_path} to {self.root}")
        try:
            with tarfile.open(output_path, "r:gz") as tar:
                tar.extractall(path=self.root)
        except tarfile.TarError as e:
            raise RuntimeError(f"Unable to extract {output_path}: {e}")
        finally:
            os.remove(output_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[int, int], int]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, (sample["class_label"], sample["bias_label"]), index

    def get_group_counts(self) -> torch.Tensor:
        group_counts = torch.zeros(self.n_groups, dtype=torch.long)
        for sample in self.samples:
            group_idx = sample["class_label"] * 2 + sample["bias_label"]
            group_counts[group_idx] += 1
        return group_counts

    def __repr__(self) -> str:
        return f"WaterbirdsDynamicRho(env={self.env}, num_samples={len(self)})"


if __name__ == "__main__":
    # === STEP 1: Generate a new metadata file ===
    # This only needs to be done once for each desired configuration.
    new_metadata_filename = "./data/waterbirds/waterbird_complete95_forest2water2/waterbirds_90.0_dynamic_metadata.csv"
    
    if not os.path.exists(new_metadata_filename):
        WaterbirdsDynamicRho.create_and_save_metadata(
            output_path=new_metadata_filename,
            strategy="borrow_from_val",
            bias_amount=0.90
        )
    else:
        print(f"Metadata file '{new_metadata_filename}' already exists. Skipping generation.")

    print("\n" + "="*50 + "\n")

    # === STEP 2: Use the new metadata file to load the dataset ===
    print("Loading dataset splits using the newly generated metadata file...")
    
    # Load the new training set
    train_set_70_borrowed = WaterbirdsDynamicRho(
        env="train",
        metadata_path=new_metadata_filename
    )
    
    # The validation set will be empty because we used the "borrow_from_val" strategy
    val_set_from_new_meta = WaterbirdsDynamicRho(
        env="val",
        metadata_path=new_metadata_filename
    )
    
    # The test set remains unchanged
    test_set_from_new_meta = WaterbirdsDynamicRho(
        env="test",
        metadata_path=new_metadata_filename
    )

    # --- Display Results ---
    datasets = {
        f"Train (from {os.path.basename(new_metadata_filename)})": train_set_70_borrowed,
        f"Val (from {os.path.basename(new_metadata_filename)})": val_set_from_new_meta,
        f"Test (from {os.path.basename(new_metadata_filename)})": test_set_from_new_meta,
    }
    
    results = []
    headers = ["Dataset", "Total Samples", "LBird/Land", "LBird/Water", "WBird/Land", "WBird/Water", "Actual Bias (LB)", "Actual Bias (WB)"]

    for name, ds in datasets.items():
        if len(ds) == 0:
            results.append([name, 0, 0, 0, 0, 0, "N/A", "N/A"])
            continue
        counts = ds.get_group_counts()
        total = len(ds)
        bias_lb = counts[0] / (counts[0] + counts[1]) if (counts[0] + counts[1]) > 0 else 0
        bias_wb = counts[3] / (counts[3] + counts[2]) if (counts[3] + counts[2]) > 0 else 0
        
        results.append([
            name, total, 
            counts[0].item(), counts[1].item(), counts[2].item(), counts[3].item(),
            f"{bias_lb:.3f}", f"{bias_wb:.3f}"
        ])
    
    print(tabulate(results, headers=headers, tablefmt="grid"))

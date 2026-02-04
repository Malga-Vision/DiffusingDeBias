

import os
import math
from typing import Dict
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidance.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidance.ModelCondition import UNet
from DiffusionFreeGuidance.Scheduler import GradualWarmupScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Datasets import *

class ConcatDatasetWithAttributes(ConcatDataset):
    """Extended ConcatDataset that maintains access to original dataset attributes"""
    @property
    def num_classes(self):
        return self.datasets[0].num_classes


def prompt_dataset_split() -> str:
    """
    Prompts the user to select the dataset configuration for `waterbirds`.
    Returns:
        str: The configuration selected by the user.
    """
    print("\nSelect the dataset configuration for Waterbirds:")
    print("1. Standard dataset (5% conflict samples)")
    print("2. Ablation: Custom split with 10%, 20%, 30%, or 40% conflict samples")
    choice = input("Select an option (1 or 2): ").strip()

    if choice == "2":
        print("\nAvailable Bias-Conflict/Align splits:")
        print("10%: 203/1827")
        print("20%: 406/1624")
        print("30%: 609/1421")
        print("40%: 812/1218")
        split = input("Enter the desired percentage (10, 20, 30, 40): ").strip()
        if split in {"10", "20", "30", "40"}:
            return f"{split}%"
        else:
            print("Invalid choice! Falling back to standard dataset.")
    return "standard"


def get_dataset(modelConfig: Dict, transform) -> ConcatDataset:
    """
    Prepares the dataset based on the configuration and user input.

    Args:
        modelConfig (Dict): Configuration dictionary.
        transform: Transformations to apply to the dataset.

    Returns:
        ConcatDataset: Combined dataset for training, validation, and testing.
    """
    if modelConfig["dataset"] == "waterbirds":
        if modelConfig["state"] == "eval":
            return Waterbirds(root=modelConfig["data_dir"], env="train", transform=transform)
        
        dataset_split = prompt_dataset_split()
        
        train_dataset = Waterbirds(root=modelConfig["data_dir"], env="train", transform=transform)
        val_dataset = Waterbirds(root=modelConfig["data_dir"], env="val", transform=transform)
        test_dataset = Waterbirds(root=modelConfig["data_dir"], env="test", transform=transform)

        if dataset_split == "standard":
            dataset = train_dataset
        else:
            conflict_count, align_count = {
                "10%": (203, 1827),
                "20%": (406, 1624),
                "30%": (609, 1421),
                "40%": (812, 1218),
            }[dataset_split]

            dataset = ConcatDatasetWithAttributes([train_dataset, val_dataset, test_dataset])

            dataset = create_balanced_dataset(
                dataset,
                align_count=align_count,
                conflict_count=conflict_count
            )

    elif modelConfig["dataset"] == "bar":
        dataset = BAR(root=modelConfig["data_dir"], env="train", transform=transform)
    elif modelConfig["dataset"] == "bffhq":
        dataset = BFFHQ(root=modelConfig["data_dir"], env="train", transform=transform)
    elif modelConfig["dataset"] == "cifar10c":
        dataset = CorruptedCIFAR10(root=modelConfig["data_dir"], split="train", percent="5pct", transform=transform)
    elif modelConfig["dataset"] == "cifar10":
        dataset = CIFAR10(root=modelConfig["data_dir"], train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ]))
    elif modelConfig["dataset"] == "imagenet9":
        dataset = ImageNet9(root=modelConfig["data_dir"], transform=transform)
    elif modelConfig["dataset"] == "urbancars":
        dataset = UrbanCars(root=modelConfig["data_dir"], split="train", transform=transform, group_label="both")
    else:
        raise ValueError(f"Dataset {modelConfig['dataset']} not supported.")

    if modelConfig["dataset"] != "cifar10":
        num_classes = dataset.num_classes
    else:
        num_classes = len(dataset.classes)

    return dataset, num_classes


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Data transform
    data_transform = A.Compose([
        A.Resize(modelConfig["img_size"], modelConfig["img_size"]),
        A.Normalize(normalization='standard'),
        A.pytorch.ToTensorV2()
    ])

    # Prepare dataset
    dataset, num_classes = get_dataset(modelConfig, data_transform)

    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=16, drop_last=False, pin_memory=True
    )

    # Calculate epochs and iterations
    iterations = modelConfig["iterations"]
    current_iteration = 0 
    epochs = iterations // len(dataloader) + 1
    print(f"\nTraining for {epochs} epochs = {iterations} iterations ({len(dataloader)} iterations per epoch).")

    # Initialize the model 
    net_model = UNet(T=modelConfig["T"], num_labels=num_classes, ch=modelConfig["channel"],
                    ch_mult=modelConfig["channel_mult"], num_res_blocks=modelConfig["num_res_blocks"], 
                    dropout=modelConfig["dropout"]).to(device)

    # Initialize optimizer 
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1
    )

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                                warm_epoch=epochs // 10, after_scheduler=cosineScheduler)
                
    # Initialize the diffusion trainer
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training       
    for epoch in range(0, epochs):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data in tqdmDataLoader:

                if modelConfig["dataset"] == "cifar10":
                    images, labels = data
                else:
                    images = data['image'] 
                    labels = data['class_label']
                
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                current_iteration += 1
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "iteration": current_iteration,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                
                if modelConfig["freq_save"] > 0 and current_iteration % modelConfig["freq_save"] == 0:
                    model_path = os.path.join(modelConfig["save_dir"], modelConfig["dataset"], "ckpt_" + str(current_iteration) + "_iterations.pt")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': net_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': warmUpScheduler.state_dict(),
                        'learning_rate': warmUpScheduler.get_last_lr(),
                        'iteration': current_iteration,
                        'epoch': epoch,
                    }, model_path)

        warmUpScheduler.step()


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: str = "cuda") -> float:

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Update with real images
    for batch in torch.split(real_images, 16):  # Process in batches to avoid memory issues
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=True)
    
    # Update with generated images
    for batch in torch.split(generated_images, 16):
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=False)
    
    return fid_metric.compute().item()


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    os.makedirs(os.path.join(modelConfig["sampled_dir"], modelConfig["dataset"]), exist_ok=True)
    
    data_transform = A.Compose([
        A.Resize(modelConfig["img_size"], modelConfig["img_size"]),
        A.Normalize(normalization='standard'),
        A.pytorch.ToTensorV2()
    ])

    # Prepare dataset
    dataset, num_classes = get_dataset(modelConfig, data_transform)

    # Load model and sampler
    model = UNet(T=modelConfig["T"], num_labels=num_classes, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                 num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    

    model_path = modelConfig["load_weights"]
    iterations = int(os.path.basename(model_path).split("_")[1])

    ckpt = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"Model loaded from {model_path}")
    del ckpt

    model.eval()
    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
    
    img_avg = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None]   
    img_std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    img_avg = img_avg.to(device)
    img_std = img_std.to(device)
    
    batch_size = modelConfig["batch_size"]
    samples = modelConfig["images_to_sample"]
    if batch_size > samples: batch_size = samples
    number_of_batches = math.ceil(samples / batch_size)
    print(f"Generating {samples} samples in {number_of_batches} batches.")

    fid_scores = []
    
    # Compute FID for each class
    for class_label in range(num_classes):
        real_images = []
        generated_images = []
        
        # Collect real images for the current class
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for real_batch in dataloader:

            if modelConfig["dataset"] == "cifar10":
                mask = real_batch[1] == class_label
                real_images.extend(real_batch[0][mask])
            else:
                mask = real_batch['class_label'] == class_label
                real_images.extend(real_batch['image'][mask])

            if len(real_images) >= samples:
                break
        real_images = torch.stack(real_images)[:samples].to(device)

        # Generate images for the current class
        for i in range(number_of_batches):
            with torch.no_grad():
                noisy_images = torch.randn(
                    size=[batch_size, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
                class_labels = torch.full((batch_size,), class_label+1, dtype=torch.long, device=device)
                sampledImgs = sampler(noisy_images, class_labels, start_t=modelConfig["T"])
                generated_images.extend(sampledImgs)

                save_image(sampledImgs * img_std + img_avg, os.path.join(modelConfig["sampled_dir"], modelConfig["dataset"], f"batch_{i}_sampled_images_class_{class_label}_iter{iterations}.png"), nrow=modelConfig["nrow"])
                print(f"Saved batch {i} class {class_label} images in {modelConfig['sampled_dir'] + modelConfig['dataset']}")
                sampledImgs = sampledImgs.cpu().detach()
                sampledImgs = sampledImgs.permute(0, 2, 3, 1).numpy()
                np.save(os.path.join(modelConfig["sampled_dir"], modelConfig["dataset"], f"batch_{i}_sampled_images_class_{class_label}.npy"), sampledImgs)


        generated_images = torch.stack(generated_images)[:samples].to(device)

        # Compute FID for the current class
        fid_score = compute_fid(real_images, generated_images, device=device)
        fid_scores.append(fid_score)
        print(f"FID score for class {class_label}: {fid_score}")

    # Compute and print the mean FID
    mean_fid_score = sum(fid_scores) / len(fid_scores)
    print(f"Mean FID score: {mean_fid_score}")
    np.savetxt(os.path.join(modelConfig["sampled_dir"], modelConfig["dataset"], f"fid_scores_{iterations}.txt"), fid_scores)

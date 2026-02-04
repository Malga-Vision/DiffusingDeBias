import os
from typing import List, Tuple, Union
import torch
from torch.nn.functional import one_hot
import random
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms
from matplotlib import pyplot as plt
from torch.utils import data
from datasets.cifar10c import CorruptedCIFAR10
from datasets.waterbirds import Waterbirds
from datasets.CMNIST import CMNIST
from datasets.bar import BAR
from datasets.BFFHQ import BFFHQ
from sklearn.metrics import classification_report
import pandas as pd
from utils.wandb_wrapper import WandbWrapper
import argparse

from erm_training import train_model_erm

import torch.nn as nn

class Hook:
    """Registers a hook at a specific layer of a network"""

    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module[1].register_forward_hook(self.hook_fn)
            self.name = module[0]
        else:
            self.hook = module[1].register_backward_hook(self.hook_fn)
            self.name = module[0]

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class FromNpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        return x, (y, y), index

    def __len__(self):
        return len(self.data)


def set_seed(seed):
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 32 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        x = self.classifier(x)
        return x


class ModelOnDiffused(torch.nn.Module):
    def __init__(self, num_classes, model, weight_path, dataset='waterbirds'):
        super().__init__()
        self.num_classes = num_classes
        self.model: nn.Module = copy.deepcopy(model)
        self.model.to("cuda")

        match dataset:
            case "cifar":
                self.transform = transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC)
            case _:
                self.transform = transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)

        self.model.load_state_dict(torch.load(weight_path))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target=None):
        self.model.eval()
        if target is None:
            return self.model(self.transform(x))
        
        with torch.no_grad():
            x = self.transform(x)
            return self.model(x), self.loss_fn(self.model(x), target)


def create_loaders():
    eval_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_set = Waterbirds(root="data", env="train", transform=eval_transform)
    val_set = Waterbirds(root="data", env="val", transform=eval_transform)
    test_set = Waterbirds(root="data", env="test", transform=eval_transform)

    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)       

    return train_loader, val_loader, test_loader, eval_transform


def create_bias_amplifier(dataset_name='waterbirds'):
    model_for_bias = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model_for_bias.classifier = torch.nn.Linear(model_for_bias.classifier.in_features, 2)

    model_for_bias.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_for_bias = model_for_bias.to("cuda")
    return model_for_bias


def read_DDPM_images(args: argparse.Namespace):
    dataset_name = args.dataset
    synth_rho = args.rho    
    address = os.path.join("data", "synthetic", "increasing_rho", "waterbirds", str(int(synth_rho)))
    print("Opening ", str(address))
    files = sorted(os.listdir(address))
    x_train = []
    y_train = []

    for i in range(0, len(files)):
        if 'npy' in files[i] and 'class_0' in files[i]:
            DDPM_images = np.load(os.path.join(address, files[i]))
            x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
            y_train.append(torch.zeros(len(DDPM_images)))

        elif 'npy' in files[i] and 'class_1' in files[i]:
            DDPM_images = np.load(os.path.join(address, files[i]))
            x_train.append(torch.from_numpy(np.moveaxis(DDPM_images, 3, 1)))
            y_train.append(torch.ones(len(DDPM_images)))
        
    x_train = torch.concatenate(x_train, dim=0)
    y_train = torch.concatenate(y_train, dim=0).long()
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ])

    dataset_DDPM = FromNpyDataset(x_train, y_train, transform=transform)
    data_loader_DDPM_train = data.DataLoader(dataset_DDPM, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

    return dataset_DDPM, data_loader_DDPM_train


@torch.no_grad()
def extract_misclassified(dataset_name: str, biased_model: nn.Module, dataset: data.Dataset, device="cuda", rho = 95, wb: WandbWrapper = None):
    dataset: Union[BAR, Waterbirds, BFFHQ, CorruptedCIFAR10]
    class_labels: List[torch.Tensor] = []
    bias_labels: List[torch.Tensor] = []
    class_predictions: List[torch.Tensor] = []
    bias_predictions: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []

    dataset.transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print(dataset.transform)

    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    x: torch.Tensor  #type-annot
    y: torch.Tensor  #type-annot
    b: torch.Tensor  #type-annot

    biased_model.eval()
    for (x, (y, b), _) in dataloader:
        x = x.to(device)
        y = y.to(device)
        b = b.to(device)

        outs: Tuple[torch.Tensor, torch.Tensor] = biased_model(x, y)
        y_preds: torch.Tensor = outs[0].argmax(dim=1)
        bias_preds: torch.Tensor = y_preds.clone()

        class_labels.append(y)
        bias_labels.append(b)
        class_predictions.append(y_preds)
        bias_predictions.append(bias_preds)
        losses.append(outs[1])

    class_labels: torch.Tensor = torch.cat(class_labels, dim=0).cpu()
    bias_labels: torch.Tensor = torch.cat(bias_labels, dim=0).cpu()
    class_predictions: torch.Tensor = torch.cat(class_predictions, dim=0).cpu()
    bias_predictions: torch.Tensor = torch.cat(bias_predictions, dim=0).cpu()
    losses: torch.Tensor = torch.cat(losses, dim=0).cpu()

    threshold = torch.mean(losses) + 3*torch.std(losses)
    print(classification_report(class_labels, class_predictions))

    outpath = os.path.join("outputs", dataset_name)
    os.makedirs(outpath, exist_ok=True)
    torch.save({
        "class_labels": class_labels,
        "bias_labels": bias_labels,
        "class_predictions": class_predictions,
        "bias_predictions": bias_predictions,
    }, os.path.join(outpath, f"{dataset_name}_biased_model_predictions.pt"))


    new_metadata_path = None
    
    bias_predictions = torch.where((class_predictions != class_labels) & (losses > threshold), 1-class_labels, class_labels)
    df = pd.read_csv("./data/waterbirds/waterbird_complete95_forest2water2/metadata.csv", header="infer",
                        index_col=0)
    train_ids = df[df["split"] == 0].index.to_numpy()
    df["ddb"] = df["place"].copy()
    df.loc[train_ids, "ddb"] = bias_predictions.numpy()
    new_metadata_path = os.path.join("outputs", "waterbirds_metadata_aug.csv")
    df.to_csv(new_metadata_path)

    csv = pd.read_csv(new_metadata_path, header="infer")
    train_set = csv[csv["split"] == 0]
    true_aligned = train_set[train_set["y"] == train_set["place"]]
    true_conflicting = train_set[train_set["y"] != train_set["place"]]

    class_0_metrics = classification_report(train_set[train_set["y"] == 0]["place"],
                                            train_set[train_set["y"] == 0]["ddb"],
                                            target_names=["Aligned", "Conflicting"])
    class_1_metrics = classification_report(train_set[train_set["y"] == 1]["place"],
                                            train_set[train_set["y"] == 1]["ddb"],
                                            target_names=["Conflicting", "Aligned"])
    aligned_metrics = classification_report(true_aligned["place"], true_aligned["ddb"],
                                            target_names=["Landbird", "Waterbird"])
    conflic_metrics = classification_report(true_conflicting["place"], true_conflicting["ddb"],
                                            target_names=["Landbird", "Waterbird"])
    overall_metrics = classification_report(train_set["place"], train_set["ddb"],
                                            target_names=["Landbird", "Waterbird"])

    print("Class 0 (Landbird)")
    print(class_0_metrics, "\n")
    print("Class 1 (Waterbird)")
    print(class_1_metrics, "\n")
    print("Bias-Conflicting", "\n")
    print(conflic_metrics, "\n")



    if wb is not None:
        wb.log_output({
            "id_metrics": 
                wb.backend.Table(dataframe=pd.DataFrame.from_dict({ 
                "class_0": class_0_metrics,
                "class_1": class_1_metrics,
                "aligned": aligned_metrics,
                "conflic": conflic_metrics,
                "overall": overall_metrics
            }, orient="index"))
        })
        metadata_file = wb.backend.Artifact("pseudolabels_metadata_file", type="dataset")
        metadata_file.add_file(new_metadata_path)
        wb.backend.log_artifact(metadata_file)

    return bias_predictions


def extract_groups(args: argparse.Namespace):
    dataset_name  = args.dataset
    retrain_model = args.retrain
    use_wb        = not args.no_wb 
    rho           = args.rho

    if dataset_name == "cifar10c":
        global TARGET_SIZE
        TARGET_SIZE = (32, 32)
    
    train_loader, val_loader, test_loader, eval_transform = create_loaders()
    config = {"dataset_name": dataset_name, }    
    train_set = Waterbirds(env="train", return_index=True)
    config["lr"] = 0.0005
    config["num_classes"] = 2
    config["epochs"] = 50
    config["wd"] = 0.01       

    biased_model = create_bias_amplifier(dataset_name)
    _, data_loader_DDPM_train = read_DDPM_images(args)

    wb = None
    if use_wb:
        wb = WandbWrapper("IncreasingRhoDiffuseDebias_0", config)

    print(train_set)
    if retrain_model:
        train_model_erm(
            biased_model,
            data_loader_DDPM_train,
            val_loader,
            None,
            "cuda",
            torch.optim.AdamW(biased_model.parameters(), lr=config["lr"], weight_decay=config["wd"]),
            num_classes=config["num_classes"],
            epochs=config["epochs"],
            wb=wb,
            name=f"biased_model_{rho}_{dataset_name}"
        )

    biased_model = ModelOnDiffused(
        config["num_classes"],
        biased_model,
        os.path.join(PATH_TO_MODELS, f"biased_model_{rho}_{dataset_name}-final.pt"),
        dataset=dataset_name
    )
    
    from erm_training import evaluate_model
    evaluate_model(biased_model, test_loader, config["num_classes"], num_bias_attributes=1, criterion=torch.nn.CrossEntropyLoss(), epoch="TEST-BA-ERM", device="cuda", wb=wb, prefix="ba-erm")

    # bias_pseudolabels: torch.Tensor = extract_misclassified(dataset_name, biased_model, train_set, "cuda", rho=rho, wb=wb)
    # print(torch.unique(torch.as_tensor(bias_pseudolabels), return_counts=True))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="waterbirds", help="dataset name. choose in [waterbirds]")
parser.add_argument("--retrain", action="store_true", help="repeat experiment and overwrite biased model, default=False")
parser.add_argument("--no_wb", action="store_false", default=False, help="disables Weights & Biases logging")
parser.add_argument("--seed", type=int, default=0, help="random state for stochastic operations")
parser.add_argument("--rho", type=float, default=95)

if __name__ == "__main__":   
    PATH_TO_MODELS = f'saved_models'
    TARGET_SIZE = (64, 64)
    args = parser.parse_args()
    set_seed(args.seed)    
    
    print(args)
    extract_groups(args)

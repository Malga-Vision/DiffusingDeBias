import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms.functional as F
import os
from extract_groups import read_DDPM_images

def show(imgs, title):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.title(title)
    plt.show()


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--dataset", default="waterbirds")


if __name__ == "__main__":
    args = parser.parse_args()
    address = os.path.join("./data/synthetic/w_1/imagenet/", args.dataset, "95")
    files = sorted(os.listdir(address))
    x_train = list()
    y_train = list()

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
    img_avg = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    img_std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x_train = (x_train * img_std) + img_avg
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    batches = len(loader)
    for i, (x, y) in loader:
        class_0 = x[y == 0]
        class_1 = x[y == 1]
        if len(class_0):
            grid_0 = torchvision.utils.make_grid(class_0)
            show(grid_0, f"batch {i+1}/{batches} class_0")
        if len(class_1):
            grid_1 = torchvision.utils.make_grid(class_1)
            show(grid_1, f"batch {i+1}/{batches} class_1")
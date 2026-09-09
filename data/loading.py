"""
Datasets used in the MDNPIC paper:

  - MHIST:        binary classification (HP vs. SSA), 224x224 H&E patches.
  - Chaoyang:     4-class colorectal tissue classification, 512x512 patches.
  - HITAFH-GCML:  6-class gastric cancer tissue classification, 224x224 patches.

All loaders return single-channel (grayscale) images resized to 224x224,
matching the input requirement of the MGCG network.
"""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMG_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


class MHISTDataset(Dataset):
    """
    MHIST (A Petri Dish for Histopathology, Wei et al. 2021).
    Expected layout of ``data_list`` (the dataset root):
        root/
          images/          # .png patches
          annotations.csv  # columns: Image Name, MajorLabel, Partition
    """
    LABEL_MAP = {"HP": 0, "SSA": 1}

    def __init__(self, data_list, train, transform=None):
        self.root = data_list
        self.transform = transform if transform is not None else (
            TRAIN_TRANSFORM if train else TEST_TRANSFORM)
        annotations = pd.read_csv(os.path.join(self.root, "annotations.csv"))
        partition = "Train" if train else "Test"
        if "Partition" in annotations.columns:
            annotations = annotations[annotations["Partition"] == partition]
        self.samples = [
            (os.path.join(self.root, "images", row["Image Name"]),
             self.LABEL_MAP[row["MajorLabel"]])
            for _, row in annotations.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class chaoyangDataset(Dataset):
    """
    Chaoyang colorectal dataset (4 tissue classes).
    Expected layout of ``data_list`` (the dataset root):
        root/
          train/  tissue1/ ... tissue4/
          test/   tissue1/ ... tissue4/
    """

    def __init__(self, data_list, train, transform=None):
        split_dir = os.path.join(data_list, "train" if train else "test")
        self.dataset = ImageFolder(split_dir,
                                   transform=transform if transform is not None else (
                                       TRAIN_TRANSFORM if train else TEST_TRANSFORM))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class HITFAH_GCMLDataset(Dataset):
    """
    HITAFH-GCML gastric cancer tissue dataset (6 tissue classes).
    Expected layout of ``data_list`` (the dataset root):
        root/
          train/  <class_name>/ ...
          test/   <class_name>/ ...
    """

    def __init__(self, data_list, train, transform=None):
        split_dir = os.path.join(data_list, "train" if train else "test")
        self.dataset = ImageFolder(split_dir,
                                   transform=transform if transform is not None else (
                                       TRAIN_TRANSFORM if train else TEST_TRANSFORM))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

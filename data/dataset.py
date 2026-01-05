import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    NormalizeIntensityd, CropForegroundd, RandFlipd, ToTensord
)

class BraTSDataset(Dataset):
    def __init__(self, json_path, phase='training', transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data[phase]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # MONAI transforms expect dictionary
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_transforms(phase='train'):
    if phase == 'train':
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            ToTensord(keys=["image", "label"])
        ])
    else:
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"])
        ])
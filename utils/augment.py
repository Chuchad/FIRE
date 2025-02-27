# based on diffusers implementation of https://pix2pixzero.github.io/

# import torch, torchvision, diffusers, transformers, PIL, types, argparse, os
import numpy as np
from typing import *
from PIL import Image
from random import random, choice
from torchvision.transforms.v2 import functional as TF
import albumentations.core.composition
import albumentations.augmentations as A

def weak_augment(im: Image):
    if random() > 0.5:
        im = TF.jpeg(im, choice([30, 100]))

    if random() > 0.5:
        sigma = 3.0 * random()
        im = TF.gaussian_blur(im, kernel_size=[9, 9], sigma=[sigma, sigma])

    return im

strong_transform = albumentations.core.composition.Compose([
        A.geometric.resize.SmallestMaxSize(max_size=512),
        A.geometric.transforms.HorizontalFlip(p=0.5),
        A.crops.transforms.RandomResizedCrop(height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.0/0.75), p=0.2),
        A.crops.RandomCrop(height=512, width=512),
        A.transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.1, p=0.8),
        A.transforms.ToGray(p=0.2),
        A.dropout.CoarseDropout(max_holes=1, min_holes=1, hole_height_range=(96, 96), hole_width_range=(96, 96), fill_value=128, p=0.2),
        A.transforms.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur()
    ])

def strong_augment(im: Image):
    im = np.array(im)
    im = strong_transform(image=im)["image"]
    im = Image.fromarray(im)
    return im

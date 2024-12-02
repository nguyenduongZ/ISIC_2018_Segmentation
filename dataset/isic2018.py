import cv2
import argparse
import os, sys
import numpy as np
from PIL import Image
from glob import glob

import torch
import albumentations as A
import torch.nn.functional as F
from torch.utils.data import Dataset

# Define root paths
__file__ = os.getcwd() + "/dataset/source"
_root = "/".join(__file__.split("/")[:-1]) + "/source/ISIC2018_Task1"
# print(_root)

def read_data(image_path, mask_path):
    """Read image and mask from the given path"""
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"Error reading image {image_path} or mask {mask_path}: {e}")
        raise  

    return image, mask

class CustomISIC2018(Dataset):
    def __init__(self, root: str = _root, split = "train", args: argparse = None):
        self.root = root
        self.args = args
        if self.args is None:
            raise ValueError("args can not be None")
        self._split = split
        self.__mode = split

        self.resize = A.Compose(
            [
                A.Resize(args.sz, args.sz),
            ]
        )

        if args.aug:
            self.augment = A.Compose([
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
                A.GridDistortion(p=1),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
                A.VerticalFlip(p=1),
                A.HorizontalFlip(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
                A.MotionBlur(p=1, blur_limit=7),
                A.MedianBlur(p=1, blur_limit=9),
                A.GaussianBlur(p=1, blur_limit=9),
                A.GaussNoise(p=1),
                A.ChannelShuffle(p=1),
                A.CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            ])
        else:
            self.augment = None  

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.images_dir = os.path.join(self.root, self._split)
        self.masks_dir = os.path.join(self.root, self._split + "_masks")
        # print(f"Images directory: {self.images_dir}")
        # print(f"Masks directory: {self.masks_dir}")

        self._images = sorted(glob(self.images_dir + "/*.jpg"))
        self._masks = sorted(glob(self.masks_dir + "/*.png"))

        if len(self._images) == 0 or len(self._masks) == 0:
            raise FileNotFoundError(f"No images or masks found in the directory.")

    @staticmethod
    def process_mask(x: torch.Tensor) -> torch.Tensor:
        uniques = torch.unique(x, sorted = True)
        if uniques.shape[0] > 3:
            x[x == 0] = uniques[2]
            uniques = torch.unique(x, sorted = True)
        for i, v in enumerate(uniques):
            x[x == v] = i
        
        x = x.to(dtype=torch.long)
        x_squeezed = x.squeeze(1)
        onehot = F.one_hot(x_squeezed, 3).permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image = self._images[idx]
        mask = self._masks[idx]

        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        if not os.path.exists(mask):
            raise FileNotFoundError(f"Mask not found: {mask}")
        
        x, y = read_data(image, mask)

        resized = self.resize(image=x, mask=y)

        if self.__mode == "train":
            transformed = self.augment(image=resized["image"], mask=resized["mask"])
            transformed_image = self.norm(image=transformed["image"])["image"]
            transformed_mask = transformed["mask"]
        else:
            transformed_image = self.norm(image=resized["image"])["image"]
            transformed_mask = resized["mask"]

        torch_img = torch.from_numpy(transformed_image).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_mask).unsqueeze(-1).permute(-1, 0, 1).float()

        target = {
            "semantic" : self.process_mask(torch_mask),
            # "category" : self._labels[idx],
        }

        return torch_img, target
    
    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, m):
        if m not in ['train', 'valid', 'test']:
            raise ValueError(f"mode can not be {m} and must be ['train', 'valid', 'test']")
        else:
            self.__mode = m
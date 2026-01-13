import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Custom Dataset Class

class CTStrokeDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(self.label_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        image = image / 255.0
        label = label / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label


# Dataset & DataLoader

train_image_dir = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\train\image"
train_label_dir = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\train\label"

train_dataset = CTStrokeDataset(train_image_dir, train_label_dir)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

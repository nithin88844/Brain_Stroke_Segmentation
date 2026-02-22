import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class StrokeVolumeDataset(Dataset):
    def __init__(self, root_dir, label_csv, num_slices=23):
        """
        root_dir  : path to train folder (contains patient folders)
        label_csv : path to labels.csv
        num_slices: fixed number of slices per volume
        """
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(label_csv)
        self.num_slices = num_slices

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get patient ID and label
        patient_id = self.labels_df.iloc[idx]["Patient_ID"]
        label = self.labels_df.iloc[idx]["Patient_Labels"]

        patient_folder = os.path.join(self.root_dir, patient_id)

        slice_files = sorted(os.listdir(patient_folder))

        slices = []

        for file in slice_files:
            img_path = os.path.join(patient_folder, file)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0  # normalize

            slices.append(image)

        slices = np.array(slices)

        # Fix slice count
        slices = self._fix_slices(slices)

        # Convert to tensor (S, 1, H, W)
        slices = torch.tensor(slices).unsqueeze(1)

        label = torch.tensor(label, dtype=torch.long)

        return slices, label

    def _fix_slices(self, slices):
        """
        Ensures every patient has same number of slices.
        """
        S = len(slices)

        if S > self.num_slices:
            # Uniform sampling
            indices = np.linspace(0, S - 1, self.num_slices).astype(int)
            slices = slices[indices]

        elif S < self.num_slices:
            # Zero padding
            pad_shape = (self.num_slices - S, *slices.shape[1:])
            pad = np.zeros(pad_shape, dtype=np.float32)
            slices = np.concatenate([slices, pad], axis=0)

        return slices


dataset = StrokeVolumeDataset(
    root_dir=r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\Transformer_Dataset\train",
    label_csv=r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\Transformer_Dataset\Patient_Labels.csv",
    num_slices=23
)

processed_data = []

for i in tqdm(range(len(dataset))):
    volume, label = dataset[i]
    processed_data.append({
        "volume": volume,
        "label": label
    })

print("Dataset processing completed.")

torch.save(processed_data, "transformer_train_dataset.pt")

print("Processed dataset saved as transformer_train_dataset.pt")
# volume, label = dataset[0]

# print("Volume shape:", volume.shape)   # Expected: (64, 1, H, W)
# print("Label:", label)

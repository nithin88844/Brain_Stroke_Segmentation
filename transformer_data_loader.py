import os
import torch
import numpy as np
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = os.listdir(feature_dir)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.feature_dir, self.files[idx])
        )
        return data["features"], data["label"]



# dataset = FeatureDataset("encoder_features")
# print(np.array(dataset[0]).shape)
import torch
from unet import *
from loss import DiceLoss, DiceBCELoss


# Model, Optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=1, out_channels=1).to(device)

criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# print(device)
# print(model.parameters())

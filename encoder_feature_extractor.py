from unet import UNet
import torch
import os
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=1, out_channels=1).to(device)
checkpoint = torch.load("checkpoints/unet_epoch_70.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# Freeze entire U-Net
for param in model.parameters():
    param.requires_grad = False

save_dir = "encoder_features"
os.makedirs(save_dir, exist_ok=True)
dataset = torch.load("transformer_train_dataset.pt")
for i in tqdm(range(len(dataset))):

    volume = dataset[i]["volume"]
    label  = dataset[i]["label"]
    volume = volume.to(device)

    features = []

    with torch.no_grad():
        for slice in volume:
            slice = slice.unsqueeze(0)  # (1, 1, H, W)
            feat = model.forward_encoder(slice)
            feat = torch.flatten(feat, start_dim=1)  # (1, D)
            features.append(feat.squeeze(0))

    features = torch.stack(features)    # (S, D)

    torch.save({
        "features": features.cpu(),
        "label": label
    }, os.path.join(save_dir, f"patient_{i}.pt"))


# processed_data = torch.load("transformer_train_dataset.pt")

# volume = processed_data[0]["volume"]
# label = processed_data[0]["label"]



# data = torch.load("processed_volumes/patient_0.pt")
# volume = data["volume"]
# label = data["label"]

# print(type(processed_data))
# print(processed_data[0]["volume"])
# print(len(processed_data))
# volume, label = processed_data[0]["volume"], processed_data[0]["label"]
# print(volume.shape)   # Expected: (S, 1, H, W)
# print(label)          # Expected: 0 or 1
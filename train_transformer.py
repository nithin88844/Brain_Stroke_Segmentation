# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import os

# from transformer_data_loader import FeatureDataset
# from transformer import StrokeTransformer

# device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = FeatureDataset("encoder_features")
# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# model = StrokeTransformer(
#     feature_dim=1024,
#     num_slices=23,
#     num_classes=2
# ).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 500

# # ✅ Create checkpoint folder
# checkpoint_dir = "transformer_model"
# os.makedirs(checkpoint_dir, exist_ok=True)

# global_step = 0   # ✅ Track total steps

# for epoch in range(num_epochs):

#     model.train()
#     epoch_loss = 0

#     progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

#     for features, labels in progress_bar:

#         features = features.to(device)
#         labels = labels.to(device)

#         preds = model(features)
#         loss = criterion(preds, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         global_step += 1   # ✅ Increment step counter

#         progress_bar.set_postfix(loss=loss.item())

#         # ✅ Save checkpoint every 50 steps
#         if global_step % 1000 == 0:
#             checkpoint_path = os.path.join(
#                 checkpoint_dir,
#                 f"transformer_step_{global_step}.pth"
#             )

#             torch.save({
#                 "epoch": epoch,
#                 "step": global_step,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "loss": loss.item()
#             }, checkpoint_path)

#             print(f"\n✅ Saved checkpoint at step {global_step}")

#     print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {epoch_loss/len(loader):.4f}")





from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from collections import Counter

from transformer_data_loader import FeatureDataset
from transformer import StrokeTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FeatureDataset("encoder_features")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# -------------------------------
# Compute class weights
# -------------------------------
all_labels = []
for _, label in dataset:
    all_labels.append(label.item())

class_counts = Counter(all_labels)
print("Class distribution:", class_counts)

total_samples = len(dataset)

weights = torch.tensor([
    total_samples / class_counts[0],
    total_samples / class_counts[1]
], dtype=torch.float32).to(device)

print("Class weights:", weights)

# -------------------------------
# Model
# -------------------------------
# model = StrokeTransformer(
#     feature_dim=1024,
#     num_slices=23,
#     num_classes=2
# ).to(device)

# criterion = nn.CrossEntropyLoss(weight=weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 500

# checkpoint_dir = "transformer_model"
# os.makedirs(checkpoint_dir, exist_ok=True)

# global_step = 0

# for epoch in range(num_epochs):

#     model.train()
#     epoch_loss = 0

#     progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

#     for features, labels in progress_bar:

#         features = features.to(device)
#         labels = labels.to(device)

#         preds = model(features)
#         loss = criterion(preds, labels)

#         optimizer.zero_grad()
#         loss.backward()

#         # Gradient clipping (optional but recommended)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()

#         epoch_loss += loss.item()
#         global_step += 1

#         progress_bar.set_postfix(loss=loss.item())

#         if global_step % 1000 == 0:
#             checkpoint_path = os.path.join(
#                 checkpoint_dir,
#                 f"transformer_step_{global_step}.pth"
#             )

#             torch.save({
#                 "epoch": epoch,
#                 "step": global_step,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "loss": loss.item()
#             }, checkpoint_path)

#             print(f"\n✅ Saved checkpoint at step {global_step}")

#     print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {epoch_loss/len(loader):.4f}")
# import os
# import cv2
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from optimizer import model, optimizer, device,criterion
# from preprocess import train_loader

# checkpoint_dir = "checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)


# num_epochs = 40

# for epoch in range(num_epochs):  
#     model.train()
#     epoch_loss = 0.0

#     progress_bar = tqdm(
#         train_loader,
#         desc=f"Epoch [{epoch+1}/{num_epochs}]",
#         leave=False
#     )

#     for step, (images, masks) in enumerate(progress_bar):
#         images = images.to(device)
#         masks = masks.to(device)

#         preds = model(images)
#         loss = criterion(preds, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#         # Update progress bar
#         progress_bar.set_postfix({
#             "step": step + 1,
#             "loss": f"{loss.item():.4f}"
#         })

#     avg_loss = epoch_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")

#     # ✅ Save checkpoint every 5 epochs
#     if (epoch + 1) % 5 == 0:
#         checkpoint_path = os.path.join(
#             checkpoint_dir,
#             f"unet_epoch_{epoch+1}.pth"
#         )
#         torch.save({
#             "epoch": epoch + 1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "loss": avg_loss
#         }, checkpoint_path)

#         print(f"✔ Checkpoint saved at: {checkpoint_path}")


import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from optimizer import model, optimizer, device, criterion
from preprocess import train_loader

# -------------------------------
# Checkpoint & loss directories
# -------------------------------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

loss_dir = "loss_logs"
os.makedirs(loss_dir, exist_ok=True)

# -------------------------------
# Training configuration
# -------------------------------
num_epochs = 40

# ✅ Store epoch-wise loss
train_epoch_losses = []

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{num_epochs}]",
        leave=False
    )

    for step, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            "step": step + 1,
            "loss": f"{loss.item():.4f}"
        })

    # ✅ BEST POSITION TO LOG LOSS (end of epoch)
    avg_loss = epoch_loss / len(train_loader)
    train_epoch_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")

    # ✅ Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"unet_epoch_{epoch+1}.pth"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, checkpoint_path)

        print(f"✔ Checkpoint saved at: {checkpoint_path}")

# -------------------------------
# ✅ Save loss values after training
# -------------------------------
loss_path = os.path.join(loss_dir, "train_epoch_losses.npy")
np.save(loss_path, np.array(train_epoch_losses))

print(f"📈 Training loss values saved at: {loss_path}")


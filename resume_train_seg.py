# import os
# import cv2
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from optimizer import model, optimizer, device,criterion
# from preprocess import train_loader

# def get_latest_checkpoint(checkpoint_dir):
#     checkpoints = [
#         os.path.join(checkpoint_dir, f)
#         for f in os.listdir(checkpoint_dir)
#         if f.endswith(".pth")
#     ]

#     if not checkpoints:
#         return None

#     checkpoints.sort(key=os.path.getmtime)
#     return checkpoints[-1]


# checkpoint_dir = "checkpoints"
# latest_ckpt = get_latest_checkpoint(checkpoint_dir)

# start_epoch = 0

# if latest_ckpt is not None:
#     print(f"🔄 Loading checkpoint: {latest_ckpt}")

#     checkpoint = torch.load(latest_ckpt, map_location=device)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     start_epoch = checkpoint["epoch"]  # resume from next epoch

# else:
#     print("⚠ No checkpoint found. Training from scratch.")



# num_epochs = 40

# for epoch in range(start_epoch,num_epochs):
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
import torch
from tqdm import tqdm
from optimizer import model, optimizer, device, criterion
from preprocess import train_loader

# ----------------------------------
# Utility: Get latest checkpoint
# ----------------------------------
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]

# ----------------------------------
# Load checkpoint (resume training)
# ----------------------------------
checkpoint_dir = "checkpoints"
latest_ckpt = get_latest_checkpoint(checkpoint_dir)

start_epoch = 40

if latest_ckpt is not None:
    print(f"🔄 Loading checkpoint: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]   # resume AFTER this epoch
    torch.cuda.empty_cache()

else:
    print("⚠ No checkpoint found. Training from scratch.")

# ----------------------------------
# Scheduler (MUST be after optimizer load)
# ----------------------------------
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4,
    verbose=True
)

# ----------------------------------
# Training configuration
# ----------------------------------
extra_epochs = 20                 # 🔥 train 20 more epochs
end_epoch = start_epoch + extra_epochs

# ----------------------------------
# Training loop
# ----------------------------------
for epoch in range(start_epoch, end_epoch):
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{end_epoch}]",
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

        progress_bar.set_postfix({
            "step": step + 1,
            "loss": f"{loss.item():.4f}"
        })

    # ----------------------------------
    # End of epoch
    # ----------------------------------
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{end_epoch}] | Avg Loss: {avg_loss:.4f}")

    # ✅ STEP SCHEDULER (CORRECT PLACE)
    scheduler.step(avg_loss)

    # ----------------------------------
    # Save checkpoint every 5 epochs
    # ----------------------------------
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


# import os
# import cv2
# import torch
# import numpy as np
# from unet import *

# def get_latest_checkpoint(checkpoint_dir):
#     checkpoints = [
#         os.path.join(checkpoint_dir, f)
#         for f in os.listdir(checkpoint_dir)
#         if f.endswith(".pth")
#     ]
#     if not checkpoints:
#         raise FileNotFoundError("No checkpoints found!")
#     checkpoints.sort(key=os.path.getmtime)
#     return checkpoints[-1]


# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = UNet(in_channels=1, out_channels=1).to(device)
# model.eval()

# checkpoint_dir = "checkpoints"
# latest_ckpt = get_latest_checkpoint(checkpoint_dir)

# checkpoint = torch.load(latest_ckpt, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])

# print(f"✅ Loaded model from: {latest_ckpt}")


# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     if image is None:
#         raise ValueError("Failed to load image")

#     image = image.astype(np.float32) / 255.0
#     image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

#     return image

# def infer_and_save(image_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     image_tensor = preprocess_image(image_path).to(device)

#     with torch.no_grad():
#         pred = model(image_tensor)
#         pred = torch.sigmoid(pred)
#         pred = (pred > 0.5).float()

#     mask = pred.squeeze().cpu().numpy() * 255
#     mask = mask.astype(np.uint8)

#     output_path = os.path.join(
#         output_dir,
#         os.path.basename(image_path)
#     )

#     cv2.imwrite(output_path, mask)
#     print(f"🧠 Segmentation saved at: {output_path}")


# if __name__ == "__main__":
#     test_image_path = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\train\image\50.png"  # Replace with your test image path
#     output_directory = "segmented_outputs"

#     infer_and_save(test_image_path, output_directory)




import os
import cv2
import torch
import numpy as np
from unet import *

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
        raise FileNotFoundError("No checkpoints found!")
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]

# ----------------------------------
# Dice Score Function (Metric)
# ----------------------------------
def dice_score(pred, target, smooth=1e-6):
    """
    pred   : predicted mask tensor (B, 1, H, W)
    target : ground truth mask     (B, 1, H, W)
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()

# ----------------------------------
# Device & model loading
# ----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=1, out_channels=1).to(device)
model.eval()

checkpoint_dir = "checkpoints"
latest_ckpt = get_latest_checkpoint(checkpoint_dir)

checkpoint = torch.load(latest_ckpt, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

print(f"✅ Loaded model from: {latest_ckpt}")

# ----------------------------------
# Preprocessing (same as training)
# ----------------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Failed to load image")

    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    return image

# ----------------------------------
# Inference + Dice computation
# ----------------------------------
def infer_and_save(image_path, mask_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_tensor = preprocess_image(image_path).to(device)
    gt_mask = preprocess_image(mask_path).to(device)

    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    # Dice score
    dice = dice_score(pred, gt_mask)

    # Save predicted mask
    mask = pred.squeeze().cpu().numpy() * 255
    mask = mask.astype(np.uint8)

    output_path = os.path.join(
        output_dir,
        os.path.basename(image_path)
    )

    cv2.imwrite(output_path, mask)

    print(f"🧠 Segmentation saved at: {output_path}")
    print(f"📊 Dice Score: {dice:.4f}")

# ----------------------------------
# Run inference
# ----------------------------------
if __name__ == "__main__":
    test_image_path = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\train\image\50.png"
    test_mask_path  = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\ct_images_dataset\train\label\50.png"

    output_directory = "segmented_outputs"

    infer_and_save(test_image_path, test_mask_path, output_directory)

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from unet import UNet
from transformer import StrokeTransformer  # your transformer file

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load U-Net
# -------------------------------
unet = UNet(in_channels=1, out_channels=1).to(device)
unet_ckpt = torch.load("checkpoints/unet_epoch_70.pth", map_location=device)
unet.load_state_dict(unet_ckpt["model_state_dict"])
unet.eval()

# Freeze
for p in unet.parameters():
    p.requires_grad = False

# -------------------------------
# Add encoder-only forward
# -------------------------------
# def forward_encoder(model, x):
#     e1 = model.enc1(x)
#     e2 = model.enc2(model.pool(e1))
#     e3 = model.enc3(model.pool(e2))
#     e4 = model.enc4(model.pool(e3))
#     b  = model.bottleneck(model.pool(e4))
#     return b

# -------------------------------
# Load Transformer
# -------------------------------
transformer = StrokeTransformer(
    feature_dim=1024,
    num_slices=23,
    num_classes=2
).to(device)

checkpoint_transformer = torch.load(r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\transformer_model\transformer_step_10000.pth", map_location=device)
transformer.load_state_dict(checkpoint_transformer["model_state_dict"])

transformer.eval()

# -------------------------------
# Inference Function
# -------------------------------
def infer_patient(patient_folder):

    output_folder = patient_folder + "_output"
    os.makedirs(output_folder, exist_ok=True)

    slice_files = sorted(os.listdir(patient_folder))

    features = []

    for file in tqdm(slice_files):

        img_path = os.path.join(patient_folder, file)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to divisible-by-16 size
        image = cv2.resize(image, (512, 512))   # 🔥 IMPORTANT

        image = image.astype(np.float32) / 255.0
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():

            # --- Segmentation ---
            seg = unet(image_tensor)
            seg = torch.sigmoid(seg)
            seg = (seg > 0.5).float()

            seg_np = seg.squeeze().cpu().numpy() * 255
            seg_np = seg_np.astype(np.uint8)

            cv2.imwrite(
                os.path.join(output_folder, file),
                seg_np
            )

            # --- Encoder Feature ---
            feat = unet.forward_encoder(image_tensor)

            # Global average pooling
            feat = torch.mean(feat, dim=[2, 3])  # (1, 1024)

            features.append(feat.squeeze(0))

    # Stack features
    features = torch.stack(features).unsqueeze(0)  # (1, 23, 1024)

    # Transformer prediction
    with torch.no_grad():
        logits = transformer(features.to(device))
        pred = torch.argmax(logits, dim=1).item()

    return pred

if __name__ == "__main__":
    folder = r"D:\Brain_Stroke_Classification_Segmentation\Brain_Stroke_App\workspace\Inputs\Patient_005"
    prediction = infer_patient(folder)

    print("Final Classification:", prediction)
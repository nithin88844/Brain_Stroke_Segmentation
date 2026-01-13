import torch
import torch.nn as nn

# Dice Loss
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        return 1 - (2 * intersection + smooth) / (union + smooth)


# Dice + BCEWithLogits Loss (Recommended)
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

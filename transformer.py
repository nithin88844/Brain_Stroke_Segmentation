import torch.nn as nn
import torch

class StrokeTransformer(nn.Module):
    def __init__(self, feature_dim=1024, num_slices=23, num_classes=2):
        super().__init__()

        # Learnable positional embedding for 23 slices
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_slices, feature_dim)
        )

        # Slightly lighter transformer (good for small dataset)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=1024,   # reduced from 2048
            dropout=0.2,            # slightly higher dropout for regularization
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3            # reduced from 4
        )

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)  # now 2
        )

    def forward(self, x):
        # x shape: (B, 23, 1024)

        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)

        # Global average pooling across slices
        x = x.mean(dim=1)

        return self.classifier(x)

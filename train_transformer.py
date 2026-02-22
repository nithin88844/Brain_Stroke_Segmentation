from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformer_data_loader import FeatureDataset
from transformer import StrokeTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FeatureDataset("encoder_features")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = StrokeTransformer(
    feature_dim=1024,
    num_slices=23,
    num_classes=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        preds = model(features)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(loader):.4f}")


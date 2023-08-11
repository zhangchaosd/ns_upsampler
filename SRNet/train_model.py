import os
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()

        # Initial upsample
        self.init_upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)

        # Residual layer
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(3)

        # Final output layer
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x = transforms.functional.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        x = self.init_upsample(x)

        # Feature extraction
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))

        # Residual connection
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)) + x2)

        # Final output
        o = self.conv4(x3)

        return o


class NSSRDataset(Dataset):
    def __init__(self, folder_path="SRNet/hr_frames"):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path)]
        self.transform2 = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path)
        img_lr = transforms.functional.resize(
            img, (1080 // 2, 1920 // 2), Image.NEAREST
        )

        img_hr = self.transform2(img)
        img_lr = self.transform2(img_lr)

        return img_lr, img_hr


def export_ONNX(model):
    import torch.onnx

    model.eval().to("cpu")
    dummy_input = torch.randn((1, 3, 1080, 1920), requires_grad=True)

    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "SRNet.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
    )
    print("Model has been converted to ONNX")


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    epochs = 50
    batch_size = 4
    lr = 1e-3
    dataset = NSSRDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = SRNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataloader, model, loss_fn, optimizer, device)
        torch.save(model.state_dict(), "SRNet_weights.pth")
    print("Done!")
    export_ONNX(model)


if __name__ == "__main__":
    main()

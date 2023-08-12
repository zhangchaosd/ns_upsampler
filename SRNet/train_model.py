import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import cv2
class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.net = SRNet()
        # self.net = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.alpha_channel = torch.ones((1, 2160, 3840), dtype=torch.uint8) * 255

    def forward(self, input_tensor):
        if not self.training:
            # BGRA (h, w, 4) uint8 to BGR (1, 3, h, w) float
            input_tensor=input_tensor[:,:,:3].float()/255.
            input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
        output = self.net(input_tensor)  # all bga
        if not self.training:
            # out is BGRA uint8 too
            # (1, 3, h, w) float to (3, h, w) uint8
            output = (output.squeeze(0) * 255.).to(torch.uint8)
            # vflip for SoftwareBitmap
            output = transforms.functional.vflip(output)
            # bgr to bgra
            output = torch.cat([output,self.alpha_channel])
            # (3, h, w) to (h, w, 3)
            output = output.permute(1, 2, 0)
        return output

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
        img_hr = cv2.imread(img_path)  # bgr hwc
        img_hr = self.transform2(img_hr)  # chw
        img_lr = transforms.functional.resize(
            img_hr, (1080 // 2, 1920 // 2), InterpolationMode.NEAREST
        )
        return img_lr, img_hr


def export_ONNX(model, name = "SRNet.onnx"):
    import torch.onnx

    model.eval().to("cpu")
    dummy_input = torch.ones((1080, 1920, 4), dtype=torch.uint8)

    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        name,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
    )
    print("Model has been converted to ONNX")


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    aver_loss = 0.
    model.train()
    for batch, (img_lr, img_hr) in enumerate(dataloader):
        img_lr, img_hr = img_lr.to(device), img_hr.to(device)

        # Compute prediction error
        pred = model(img_lr)
        loss = loss_fn(pred, img_hr)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        aver_loss += loss.item()
        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(img_lr)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    aver_loss /= num_batches
    print(f"Aver loss: {aver_loss:>7f}")  #  nn.Upsample is 0.0006


def main(epochs = 10, lr = 0.001):
    batch_size = 16
    dataset = NSSRDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = Adapter().to(device)
    model_params = torch.load("SRNet_weights.pth",map_location="cpu")
    model.load_state_dict(model_params)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataloader, model, loss_fn, optimizer, device)
        torch.save(model.state_dict(), "SRNet_weights.pth")
    print("Done!")
    export_ONNX(model)


if __name__ == "__main__":
    main(10, 0.0005)
    # model = Adapter()
    # p = torch.load("SRNet_weights.pth",map_location="cpu")
    # model.load_state_dict(p)
    # export_ONNX(model, "SRNet7.onnx")
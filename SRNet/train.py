import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FRNet, export
from dataset import NSSRDataset


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    aver_loss = 0.
    model.train()
    for batch, (img_lr, img_hr) in enumerate(dataloader):
        img_lr, img_hr = img_lr.to(device), img_hr.to(device)
        pred = model.forward_sequence(img_lr)

        loss = loss_fn(pred, img_hr)
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
    batch_size = 4
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
    model = FRNet().to(device)
    model_params = torch.load("SRNet_weights0830.pth",map_location="cpu")
    model.load_state_dict(model_params)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataloader, model, loss_fn, optimizer, device)
        torch.save(model.state_dict(), "SRNet_weights.pth")
    print("Done!")
    export(model)


if __name__ == "__main__":
    main(10, 0.0005)

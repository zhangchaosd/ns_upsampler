import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SRDataset(Dataset):
    def __init__(self, hr_path, lr_path):
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.videos = [video for video in os.listdir(hr_path) if video.endswith("mp4")]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        hr_video_path = os.path.join(self.hr_path, self.videos[idx])
        random_14 = random.randint(0, 3)
        lr_video_path = os.path.join(
            self.lr_path, self.videos[idx][:-4] + f"_{random_14}.mp4"
        )
        hr_frames = self.read_video(hr_video_path)
        lr_frames = self.read_video(lr_video_path)
        # tchw
        return dict(lr=lr_frames, gt=hr_frames)

    def read_video(self, video_path):
        print(video_path)
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.from_numpy(frame).float()
            frame /= 255.0
            frames.append(frame.unsqueeze(0))
        cap.release()
        return torch.cat(frames, dim=0)


def create_dataloader(opt):
    dataset = SRDataset(opt["dataset"]["hr_path"], opt["dataset"]["lr_path"])
    dataloader = DataLoader(
        dataset, batch_size=opt["dataset"]["batch_size"], shuffle=True, pin_memory=True
    )
    return dataloader

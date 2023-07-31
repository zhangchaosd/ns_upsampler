import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SRDataset(Dataset):
    def __init__(self, hr_path, lr_path, crop=128):
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.videos = [video for video in os.listdir(hr_path) if video.endswith("mp4")]
        self.crop_size = crop
        self.scale = 2

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
        hr_frames, lr_frames = self.crop_sequence(hr_frames, lr_frames)
        return dict(lr=lr_frames, gt=hr_frames)

    def read_video(self, video_path):
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
            if len(frames) > 5:
                break
        cap.release()
        return torch.cat(frames, dim=0)

    def crop_sequence(self, gt_frms, lr_frms):
        gt_csz = 128
        lr_csz = 128 // self.scale

        lr_h, lr_w = lr_frms.shape[-2:]
        assert (lr_csz <= lr_h) and (
            lr_csz <= lr_w
        ), "the crop size is larger than the image size"

        # crop lr
        lr_top = random.randint(0, lr_h - lr_csz)
        lr_left = random.randint(0, lr_w - lr_csz)
        lr_pats = lr_frms[..., lr_top : lr_top + lr_csz, lr_left : lr_left + lr_csz]

        # crop gt
        gt_top = lr_top * self.scale
        gt_left = lr_left * self.scale
        gt_pats = gt_frms[..., gt_top : gt_top + gt_csz, gt_left : gt_left + gt_csz]

        return gt_pats, lr_pats


def create_dataloader(opt):
    dataset = SRDataset(
        opt["dataset"]["hr_path"],
        opt["dataset"]["lr_path"],
        opt["dataset"]["crop_size"],
    )
    dataloader = DataLoader(
        dataset, batch_size=opt["dataset"]["batch_size"], shuffle=True, pin_memory=True
    )
    return dataloader

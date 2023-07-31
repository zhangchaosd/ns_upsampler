import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.networks.egvsr_nets import FRNet


def parse_video(path, scale, model, device):
    video_reader = cv2.VideoCapture(path)
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = frame_width * scale
    h = frame_height * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_saver = cv2.VideoWriter(path[:-4] + "_new.mp4", fourcc, 30.0, (w, h))
    frame_count = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    # while video_reader.isOpened():
    for _ in tqdm(range(int(frame_count))):
        ret, frame = video_reader.read()
        if not ret:
            break
        start_time = time.time()
        # frame = Image.fromarray(frame)
        # frame = frame.filter(ImageFilter.BoxBlur(radius=0.05))
        # frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.from_numpy(frame).float()
        frame /= 255.0
        # chw
        frame = frame.unsqueeze(0)
        frame = model.deliver_frame(frame, device)
        frame = frame.numpy()
        frame = np.clip(frame, 0.0, 1.0)
        frame *= 255
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        end_time = time.time()

        # Calculate elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        print("Cost time: ", elapsed_time_ms)
        video_saver.write(frame)  # hwc

    video_reader.release()
    video_saver.release()


if __name__ == "__main__":
    device = "mps"
    device = torch.device(device)
    scale = 2
    model = FRNet(scale=scale)
    s = torch.load("debug/train/ckpt/G_iter20000.pth", map_location=device)
    res = model.load_state_dict(s, strict=False)
    model.eval()
    model.to(device)
    parse_video("v1.mp4", scale, model, device)
    # parse_img("/Users/z/Downloads/IMG_1657.PNG")

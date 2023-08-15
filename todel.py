import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import cv2


# shuffle_channels = transforms.v2.RandomChannelPermutation()
transform2 = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

img_path = "test1.png"
img_hr = cv2.imread(img_path)  # bgr hwc
print(img_hr.shape)

img_hr = transform2(img_hr)  # chw
img_lr = transforms.functional.resize(
    img_hr, (1080 // 2, 1920 // 2), InterpolationMode.NEAREST
)
print(img_hr.shape, img_lr.shape)
permute_order = torch.randperm(3)
print(permute_order)
# 使用permute重新排列通道
x_permuted = img_hr[permute_order]

print(img_hr.shape, img_lr.shape)
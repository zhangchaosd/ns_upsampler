import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor, resize
from PIL import Image
import random
class NSSRDataset(Dataset):
    def __init__(self, folder_path="SRNet/hr_frames", crop_size=256, num_frames=16):
        self.folder_path = folder_path
        self.crop_size = crop_size
        self.num_frames = num_frames
        image_files = [f for f in os.listdir(folder_path)]
        data_dict = {}
        for item in image_files:
            video_name, frame_num = item.split("_frame_")
            if video_name not in data_dict:
                data_dict[video_name] = []
            data_dict[video_name].append(frame_num)

        self.image_files = []
        for video_name, frame_num_list in data_dict.items():
            sorted_frame_num_list = sorted(frame_num_list)
            assert len(sorted_frame_num_list) >= num_frames
            self.image_files.append([f"{video_name}_frame_{frame_num}" for frame_num in sorted_frame_num_list])

    def __len__(self):
        return len(self.image_files) * 100
        # return len(self.image_files) * (len(self.image_files[0]) - self.num_frames)

    def __getitem__(self, idx):
        sublist = random.choice(self.image_files)
        start_index = random.randint(0, len(sublist) - self.num_frames)
        imgs_list = sublist[start_index:start_index + self.num_frames]
        return self.open_random_crop(imgs_list)  # tchw

    def open_random_crop(self, imgs_list):
        hr_imgs = []
        lr_imgs = []
        lr_size = (self.crop_size // 2, self.crop_size // 2)
        for image_path in imgs_list:
            with Image.open(os.path.join(self.folder_path, image_path)) as img:
                w, h = img.size
                left = random.randint(0, w - self.crop_size)
                upper = random.randint(0, h - self.crop_size)
                right = left + self.crop_size
                lower = upper + self.crop_size
                cropped_img = img.crop((left, upper, right, lower))
                hr_img = to_tensor(cropped_img)
                hr_imgs.append(hr_img.unsqueeze(0))
                lr_imgs.append(resize(hr_img, lr_size, InterpolationMode.NEAREST).unsqueeze(0))
        hr_imgs = torch.cat(hr_imgs, dim=0)
        lr_imgs = torch.cat(lr_imgs, dim=0)
        return lr_imgs, hr_imgs


# dataset = NSSRDataset(crop_size=256)
# a, b = dataset.__getitem__(10)
# print(a.shape, b.shape)
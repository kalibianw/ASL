from utils import Config

from torch.utils.data import Dataset
from torchvision.io import read_image

import torch

from tqdm import tqdm


class NumberOfFileNotSame(Exception):
    def __init__(self):
        super().__init__("The number of image files and label files doesn't same.")


class CustomImageDataset(Dataset):
    def __init__(self, cfg: Config, dataset_df, resize_transform=None):
        self.resize_transform = resize_transform

        self.cfg = cfg

        self.total_img_file_paths = dataset_df["img_file_paths"]
        self.labels = dataset_df["labels"]
        self.lndmrks = dataset_df["lndmrks"]
        if len(self.total_img_file_paths) != len(self.labels):
            raise NumberOfFileNotSame()

    def __len__(self):
        return len(self.total_img_file_paths)

    def __getitem__(self, idx):
        x = read_image(self.total_img_file_paths[idx])
        org_img_size = x.size()[1:]

        y_label = torch.Tensor([self.labels[idx]])
        y_lndmrk = torch.Tensor(self.lndmrks[idx])

        if self.resize_transform is not None:
            x = self.resize_transform(x)
            new_lndmrk_x_coord = y_lndmrk[:, 0] / org_img_size[1] * self.cfg.output_hm_shape[1]
            new_lndmrk_y_coord = y_lndmrk[:, 1] / org_img_size[0] * self.cfg.output_hm_shape[0]
            y_lndmrk = torch.stack((new_lndmrk_x_coord, new_lndmrk_y_coord), dim=1)
        x = x.to(torch.float32)

        return x, y_label, y_lndmrk


class CustomImageDatasetLoadAllIntoMemory(Dataset):
    def __init__(self, cfg: Config, dataset_df, resize_transform=None):
        self.cfg = cfg

        self.total_img_file_paths = dataset_df["img_file_paths"]
        if len(self.total_img_file_paths) != len(dataset_df["labels"]):
            raise NumberOfFileNotSame

        self.imgs = list()
        self.labels = list()
        self.lndmrks = list()
        load_image_tqdm = tqdm(
            enumerate(zip(self.total_img_file_paths, dataset_df["labels"], dataset_df["lndmrks"])),
            desc=f"Load images...",
            total=len(self.total_img_file_paths)
        )
        for i, (img_file_path, label, lndmrk) in load_image_tqdm:
            img = read_image(img_file_path)
            org_img_size = img.size()[1:]

            y_label = torch.Tensor([dataset_df["labels"][i]])
            y_lndmrk = torch.Tensor(dataset_df["lndmrks"][i])

            if resize_transform is not None:
                img = resize_transform(img)
                new_lndmrk_x_coord = y_lndmrk[:, 0] / org_img_size[1] * self.cfg.output_hm_shape[1]
                new_lndmrk_y_coord = y_lndmrk[:, 1] / org_img_size[0] * self.cfg.output_hm_shape[0]
                y_lndmrk = torch.stack((new_lndmrk_x_coord, new_lndmrk_y_coord), dim=1)

            img = img.to(torch.float32)

            self.imgs.append(img)
            self.labels.append(y_label)
            self.lndmrks.append(y_lndmrk)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y_label, y_lndmrk = self.imgs[idx], self.labels[idx], self.lndmrks[idx]

        return x, y_label, y_lndmrk

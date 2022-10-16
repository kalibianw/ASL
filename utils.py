from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt


class config:
    dataset_root_path = "D:/AI/data/ASL Alphabet Synthetic/dataset/"
    output_hm_shape = (256, 256)  # (height, width)
    sigma = 2.5


class NumberOfFileNotSame(Exception):
    def __init__(self):
        super().__init__("The number of image files and label files doesn't same.")


class Visualization:
    def __init__(self, cfg: config):
        self.cfg = cfg

    def single_heatmap_visualization(self, heatmap_tensor):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        heatmap += heatmap_tensor.cpu().numpy()

        plt.imshow(heatmap)
        plt.show()

    def multi_heatmap_visualization(self, heatmaps_tensor):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        for joint_heatmap in heatmaps_tensor.cpu().numpy():
            heatmap = heatmap + joint_heatmap

        plt.imshow(heatmap)
        plt.show()


class CustomImageDataset(Dataset):
    def __init__(self, cfg: config, transform=None):
        self.transform = transform

        self.cfg = cfg

        self.total_img_file_paths = list()
        self.total_label_file_paths = list()
        self.dir_names = list()
        for dir_name in os.listdir(self.cfg.dataset_root_path):
            self.dir_names.append(dir_name)
            dir_path = os.path.join(self.cfg.dataset_root_path, dir_name)
            img_dir_path = os.path.join(dir_path, "lit")
            anno_dir_path = os.path.join(dir_path, "annotation")

            for img_name in os.listdir(os.path.join(dir_path, "lit")):
                img_file_path = os.path.join(img_dir_path, img_name)
                self.total_img_file_paths.append(img_file_path)

            for anno_name in os.listdir(os.path.join(dir_path, "annotation")):
                label_file_path = os.path.join(anno_dir_path, anno_name)
                self.total_label_file_paths.append(label_file_path)
        if len(self.total_img_file_paths) != len(self.total_label_file_paths):
            raise NumberOfFileNotSame()

    def __len__(self):
        return len(self.total_img_file_paths)

    def __getitem__(self, idx):
        x = read_image(self.total_img_file_paths[idx])
        org_img_size = x.size()[1:]

        df = pd.read_json(self.total_label_file_paths[idx])
        y_label = torch.Tensor([self.dir_names.index(df["Letter"][0])])
        y_lndmrk = torch.Tensor(df["Landmarks"])

        if self.transform is not None:
            x = self.transform(x)
            new_lndmrk_x_coord = y_lndmrk[:, 0] / org_img_size[1] * self.cfg.output_hm_shape[1]
            new_lndmrk_y_coord = y_lndmrk[:, 1] / org_img_size[0] * self.cfg.output_hm_shape[0]
            y_lndmrk = torch.stack((new_lndmrk_x_coord, new_lndmrk_y_coord), dim=1)
        x = x.to(torch.float32)

        return x, y_label, y_lndmrk


class CustomImageDatasetLoadAllIntoMemory(Dataset):
    def __init__(self, cfg: config, transform=None):
        self.cfg = cfg

        total_img_file_paths = list()
        total_label_file_paths = list()
        self.dir_names = list()
        for dir_name in os.listdir(self.cfg.dataset_root_path):
            self.dir_names.append(dir_name)
            dir_path = os.path.join(self.cfg.dataset_root_path, dir_name)
            img_dir_path = os.path.join(dir_path, "lit")
            anno_dir_path = os.path.join(dir_path, "annotation")

            for img_name in os.listdir(os.path.join(dir_path, "lit")):
                img_file_path = os.path.join(img_dir_path, img_name)
                total_img_file_paths.append(img_file_path)

            for anno_name in os.listdir(os.path.join(dir_path, "annotation")):
                label_file_path = os.path.join(anno_dir_path, anno_name)
                total_label_file_paths.append(label_file_path)
        if len(total_img_file_paths) != len(total_label_file_paths):
            raise NumberOfFileNotSame

        self.imgs = list()
        self.labels = list()
        self.lndmrks = list()
        load_image_tqdm = tqdm(
            enumerate(zip(total_img_file_paths, total_label_file_paths)),
            desc=f"Load images...",
            total=len(total_img_file_paths)
        )
        cnt = 0
        for i, (img_file_path, label_file_path) in load_image_tqdm:
            img = read_image(img_file_path)
            org_img_size = img.size()[1:]

            df = pd.read_json(label_file_path)
            y_label = torch.Tensor([self.dir_names.index(df["Letter"][0])])
            y_lndmrk = torch.Tensor(df["Landmarks"])

            if transform is not None:
                img = transform(img)
                new_lndmrk_x_coord = y_lndmrk[:, 0] / org_img_size[1] * self.cfg.output_hm_shape[1]
                new_lndmrk_y_coord = y_lndmrk[:, 1] / org_img_size[0] * self.cfg.output_hm_shape[0]
                y_lndmrk = torch.stack((new_lndmrk_x_coord, new_lndmrk_y_coord), dim=1)

            img = img.to(torch.float32)

            self.imgs.append(img)
            self.labels.append(y_label)
            self.lndmrks.append(y_lndmrk)

            cnt += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y_label, y_lndmrk = self.imgs[idx], self.labels[idx], self.lndmrks[idx]

        return x, y_label, y_lndmrk


class Model(nn.Module):
    def __init__(self, cfg: config):
        super(Model, self).__init__()
        self.cfg = cfg

    def render_gaussian_heatmap(self, joint_coords: torch.Tensor):
        heatmaps = list()
        for joint_coord in joint_coords:
            x = torch.arange(self.cfg.output_hm_shape[1])
            y = torch.arange(self.cfg.output_hm_shape[0])
            yy, xx = torch.meshgrid(y, x)
            xx = xx[:, :].cuda().float()
            yy = yy[:, :].cuda().float()

            x = joint_coord[0].item()
            y = joint_coord[1].item()

            heatmap = torch.exp(-(((xx - x) / self.cfg.sigma) ** 2) / 2 - (((yy - y) / self.cfg.sigma) ** 2) / 2)
            heatmap = heatmap * 255
            heatmaps.append(heatmap)

        return heatmaps
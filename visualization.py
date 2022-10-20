from utils import Config, Visualization
from model import Model
from dataset import CustomImageDataset, CustomImageDatasetLoadAllIntoMemory

from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def main():
    cfg = Config()

    train_df = pd.read_json(f"{os.path.dirname(cfg.dataset_root_path)}/train_dataset.json")

    resize_transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    cid = CustomImageDataset(
        cfg=cfg,
        img_file_paths=train_df["train_img_file_paths"],
        labels=train_df["train_labels"],
        lndmrks=train_df["train_lndmrks"],
        resize_transform=resize_transform
    )

    loader = DataLoader(cid, batch_size=32, shuffle=True)

    x, y_label, y_lndmrk = next(iter(loader))

    model = Model(cfg=cfg)
    heatmaps_tensor = model.render_gaussian_heatmap(joint_coords=y_lndmrk[0])

    stacked_heatmaps = torch.stack(heatmaps_tensor)

    vis = Visualization(cfg=cfg)
    vis.multi_heatmap_visualization(heatmaps_tensor=stacked_heatmaps)

    print(y_label[0])
    plt.imshow(x[0][0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    main()

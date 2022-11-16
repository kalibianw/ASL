from utils import Config, read_json, Visualization
from model import Model
from dataset import CustomImageDataset, CustomImageDatasetLoadAllIntoMemory

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import os


def main():
    cfg = Config()
    cfg.batch_size = 1

    train_df = read_json(f"{os.path.dirname(cfg.dataset_root_path)}/train_dataset.json")

    resize_transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    cid = CustomImageDataset(
        cfg=cfg,
        dataset_df=train_df,
        resize_transform=resize_transform
    )

    loader = DataLoader(cid, batch_size=cfg.batch_size, shuffle=True)

    x, y_label, y_lndmrk = next(iter(loader))

    model = Model(cfg=cfg)
    heatmaps_tensor = model.render_gaussian_heatmap(joint_coords=y_lndmrk[0])

    stacked_heatmaps = torch.stack(heatmaps_tensor)

    vis = Visualization(cfg=cfg)
    vis.multiple_joint_heatmap_visualization(
        heatmaps_tensor=stacked_heatmaps,
        title="Ground Truth Heatmap",
        export_fig_name="plot/GT Heatmap"
    )

    img = x[0].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8).copy()

    plt.imshow(img)
    plt.title("Original")
    plt.tight_layout()
    plt.savefig(
        "plot/Original Image.png",
        dpi=300
    )
    plt.show()

    img = vis.draw_line(
        img,
        lndmrk=y_lndmrk[0].cpu().numpy(),
        color=(255, 255, 255)
    )
    plt.imshow(img)
    plt.title(f"Skeleton")
    plt.tight_layout()
    plt.savefig(
        "plot/GT Skeleton.png",
        dpi=300
    )
    plt.show()


if __name__ == '__main__':
    main()

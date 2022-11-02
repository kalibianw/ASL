from utils import Config, read_json, Visualization
from model import Model
from dataset import CustomImageDataset, CustomImageDatasetLoadAllIntoMemory

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import os


def main():
    cfg = Config()

    train_df = read_json(f"{os.path.dirname(cfg.dataset_root_path)}/train_dataset.json")

    resize_transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    cid = CustomImageDataset(
        cfg=cfg,
        dataset_df=train_df,
        resize_transform=resize_transform
    )

    loader = DataLoader(cid, batch_size=32, shuffle=True)

    x, y_label, y_lndmrk = next(iter(loader))

    model = Model(cfg=cfg)
    heatmaps_tensor = model.render_gaussian_heatmap(joint_coords=y_lndmrk[0])

    stacked_heatmaps = torch.stack(heatmaps_tensor)

    vis = Visualization(cfg=cfg)
    vis.multiple_joint_heatmap_visualization(heatmaps_tensor=stacked_heatmaps)

    print(y_label[0])
    plt.imshow(x[0][0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    main()

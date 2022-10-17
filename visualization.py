import torch

from utils import Config, Visualization
from model import Model
from dataset import CustomImageDataset, CustomImageDatasetLoadAllIntoMemory

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

import matplotlib.pyplot as plt


def main():
    cfg = Config()

    transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    cid = CustomImageDataset(cfg=cfg, transform=transform)

    loader = DataLoader(cid, batch_size=1, shuffle=True)

    x, y_label, y_lndmrk = next(iter(loader))

    model = Model(cfg=cfg, resnet_type=18)
    heatmaps_tensor = model.render_gaussian_heatmap(joint_coords=y_lndmrk[0])

    stacked_heatmaps = torch.stack(heatmaps_tensor)

    vis = Visualization(cfg=cfg)
    vis.multi_heatmap_visualization(heatmaps_tensor=stacked_heatmaps)

    plt.imshow(x[0][0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    main()

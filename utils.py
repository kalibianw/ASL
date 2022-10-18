import torch

import numpy as np
import matplotlib.pyplot as plt


class Config:
    dataset_root_path = "D:/AI/data/ASL Alphabet Synthetic/dataset/"
    output_hm_shape = (256, 256)  # (height, width)
    joint_num = 26
    sigma = 2.5
    resnet_type = 18    # 18, 34, 50, 101, 152


class Visualization:
    def __init__(self, cfg: Config):
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


def export_model(model, dsize, device, export_name="model.onnx"):
    dummy_data = torch.empty(
        size=dsize,
        device=device
    )
    torch.onnx.export(
        model,
        dummy_data,
        export_name
    )

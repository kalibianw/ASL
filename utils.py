import pandas as pd
import torch

import numpy as np
import matplotlib.pyplot as plt
import json


class Config:
    # for data
    dataset_root_path = "D:/AI/data/ASL Alphabet Synthetic/dataset"
    output_hm_shape = (256, 256)  # (height, width)
    joint_num = 26

    # for generate gaussian heatmap
    sigma = 2.5

    # for NN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_type = 18  # 18, 34, 50, 101, 152
    batch_size = 32


class Visualization:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def single_joint_heatmap_visualization(self, heatmap_tensor):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        heatmap += heatmap_tensor.cpu().numpy()

        plt.imshow(heatmap)
        plt.show()

    def multiple_joint_heatmap_visualization(self, heatmaps_tensor):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        for joint_heatmap in heatmaps_tensor.cpu().numpy():
            heatmap = heatmap + joint_heatmap

        plt.imshow(heatmap)
        plt.show()


def read_json(path):
    with open(path) as handler:
        data = json.load(handler)
    df = pd.DataFrame(data["data"])

    return df


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

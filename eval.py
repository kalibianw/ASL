from utils import Config, read_json, Visualization
from dataset import CustomImageDataset
from exceptions import ModelTypeError

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

from onnx2torch import convert
import onnx

from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    cfg = Config()
    cfg.batch_size = 1

    model_paths = [
        "model/pt_[1.3895_0.0190]_ij.onnx"
    ]

    test_df = read_json(f"{os.path.dirname(cfg.dataset_root_path)}/test_dataset_ij.json")

    resize_transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    cid = CustomImageDataset(
        cfg=cfg,
        dataset_df=test_df,
        resize_transform=resize_transform
    )

    loader = DataLoader(cid, batch_size=cfg.batch_size, shuffle=True)

    x, y_label, y_lndmrk = next(iter(loader))
    x = x.to(cfg.device)

    for model_path in model_paths:
        model_ext = os.path.splitext(model_path)[1][1:]
        if model_ext == "onnx":
            print("Detect the onnx model")
            onnx_model = onnx.load_model(model_path)
            model = convert(onnx_model)
        elif model_ext == "pt":
            print("Detect the pt model")
            model = torch.load(model_path)
        else:
            raise ModelTypeError
        summary(
            model=model,
            input_size=x.size()
        )
        model.eval()

        heatmap_out, cls_out = model(x)

        vis = Visualization(cfg=cfg)
        vis.multiple_joint_heatmap_visualization(
            heatmaps_tensor=heatmap_out[0],
            export_fig_name=f"plot/{os.path.splitext(os.path.basename(model_path))[0]}"
        )
        plt.imshow(x[0][0].cpu().numpy())
        plt.show()

        print(y_label[0])
        print(np.argmax(cls_out[0].cpu().detach().numpy()))


if __name__ == '__main__':
    main()

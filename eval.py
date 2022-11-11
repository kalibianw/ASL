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

BATCH_SIZE = 1
# MODEL_PATH = "model/pt_[1.2928_0.3989].onnx"
MODEL_PATHS = [
    "model/pt_[1.2928_0.3989].onnx",
    "model/pt_[1.7891_0.1889].onnx",
    "model/pt_[2.7934_0.1285].onnx"
]


def main():
    cfg = Config()
    cfg.batch_size = BATCH_SIZE

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

    for MODEL_PATH in MODEL_PATHS:
        model_ext = os.path.splitext(MODEL_PATH)[1][1:]
        if model_ext == "onnx":
            print("Detect the onnx model")
            onnx_model = onnx.load_model(MODEL_PATH)
            model = convert(onnx_model)
        elif model_ext == "pt":
            print("Detect the pt model")
            model = torch.load(MODEL_PATH)
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
            export_fig_name=f"plot/{os.path.splitext(os.path.basename(MODEL_PATH))[0]}"
        )
        plt.imshow(x[0][0].cpu().numpy())
        plt.show()

        print(y_label[0])
        print(np.argmax(cls_out[0].cpu().detach().numpy()))


if __name__ == '__main__':
    main()

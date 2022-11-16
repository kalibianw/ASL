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
        "output/2022_11_14_17_43_2/model/pt_[0.9347_0.0588_0.9934]_ij.onnx",
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

        heatmap_out = heatmap_out[0].to("cpu").detach().numpy()
        cls_out = cls_out[0].to("cpu").detach().numpy()

        argmax_coord = list()
        for heatmap in heatmap_out:
            y_coord, x_coord = np.unravel_index(heatmap.argmax(), heatmap.shape)
            argmax_coord.append((x_coord, y_coord))
        print(argmax_coord)

        img = x.cpu().detach().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8).copy()

        img = vis.draw_line(img=img, lndmrk=argmax_coord, color=(255, 255, 255))
        plt.imshow(img)
        plt.title(f"{np.argmax(cls_out)}")
        plt.show()


if __name__ == '__main__':
    main()

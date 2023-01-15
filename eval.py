from utils import Config, read_json, Visualization
from dataset import CustomImageDataset
from exceptions import ModelTypeError
from model import Model
from trainer import TrainEvalModule

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

from onnx2torch import convert
import onnx

from torchinfo import summary

from sklearn.metrics import confusion_matrix

import os


def main():
    cfg = Config()

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
    for model_path in model_paths:
        model_ext = os.path.splitext(model_path)[1][1:]
        if model_ext == "onnx":
            print("Detect the onnx model")
            onnx_model = onnx.load_model(model_path)
            model = convert(onnx_model)
        elif model_ext == "pt":
            print("Detect the pt model")
            weight = torch.load(model_path)
            model = Model(cfg=cfg)
            model.load_state_dict(weight)
        else:
            raise ModelTypeError
        summary(
            model=model,
            input_size=(32, 3, 256, 256)
        )

        tem = TrainEvalModule(
            cfg,
            model=model,
            heatmap_loss=nn.MSELoss(),
            class_loss=nn.CrossEntropyLoss()
        )

        gaussian_renderer = Model(cfg)
        (cls_target, cls_output) = tem.evaluate(loader, gaussian_renderer, return_cls_output=True)[-1]
        print(cls_target)
        print(cls_output)
        print(cls_target.shape)
        print(cls_output.shape)

        vis = Visualization(cfg)

        cm = confusion_matrix(
            y_true=cls_target,
            y_pred=cls_output
        )
        vis.plot_confusion_matrix(
            cm,
            target_names=[*range(0, 25)],
            normalize=False,
            export_name="plot/confusion_matrix.png"
        )


if __name__ == '__main__':
    main()

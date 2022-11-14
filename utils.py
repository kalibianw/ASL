import pandas as pd
import numpy as np
import json

import torch

import cv2
import matplotlib.pyplot as plt

from copy import deepcopy
import itertools


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

    # for training
    lr = 1e-3
    num_epochs = 1000
    early_stopping_patience = 10
    reduce_lr_patience = 3
    reduce_lr_rate = 0.5
    heatmap_loss_rate = 1
    class_loss_rate = 1


class Visualization:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def draw_line(self, img, lndmrk, color):
        lndmrk = np.asarray(lndmrk, dtype=int)
        tmp = deepcopy(lndmrk[17:22])
        lndmrk[17:22] = lndmrk[12:17]
        lndmrk[12:17] = tmp

        for i in range(len(lndmrk)):
            if i % 5 == 3 and i < 23:
                for j in range(0, 3):
                    self._line1(img, i + j, lndmrk, color)
                    self._line2(img, i, lndmrk, color)

        # Wrist
        cv2.line(img, lndmrk[0], lndmrk[1], color, 2)
        cv2.line(img, lndmrk[1], lndmrk[3], color, 2)
        cv2.line(img, lndmrk[1], lndmrk[18], color, 2)

        # Thumb
        cv2.line(img, lndmrk[23], lndmrk[1], color, 2)
        cv2.line(img, lndmrk[23], lndmrk[24], color, 2)
        cv2.line(img, lndmrk[24], lndmrk[25], color, 2)

        return img

    def _line1(self, img, num, data, color):
        cv2.line(img, data[num], data[num + 1], color, 2)

    def _line2(self, img, num, data, color):
        cv2.line(img, data[num], data[num + 5], color, 2)

    def plot_confusion_matrix(self, cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)

        if labels:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    def single_joint_heatmap_visualization(self, heatmap_tensor, export_fig_name=""):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        heatmap += heatmap_tensor.cpu().detach().numpy()

        plt.imshow(heatmap)
        if export_fig_name != "":
            plt.savefig(
                f"{export_fig_name}.png",
                dpi=300
            )
        plt.show()

    def multiple_joint_heatmap_visualization(self, heatmaps_tensor, export_fig_name=""):
        heatmap = np.zeros(shape=self.cfg.output_hm_shape)
        for joint_heatmap in heatmaps_tensor.cpu().detach().numpy():
            heatmap = heatmap + joint_heatmap

        plt.imshow(heatmap)
        if export_fig_name != "":
            plt.savefig(
                f"{export_fig_name}.png",
                dpi=300
            )
        plt.show()


def read_json(path):
    with open(path) as handler:
        data = json.load(handler)
    df = pd.DataFrame(data["data"])

    return df


def export_model(model, input_size, device, export_onnx=True, export_name="model"):
    torch.save(
        obj=model.state_dict(),
        f=f"{export_name}.pt"
    )

    if export_onnx:
        dummy_data = torch.empty(
            size=input_size,
            device=device
        )
        torch.onnx.export(
            model,
            dummy_data,
            f"{export_name}.onnx"
        )

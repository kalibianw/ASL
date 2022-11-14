from utils import Config, read_json, export_model
from dataset import CustomImageDatasetLoadAllIntoMemory
from model import Model
from trainer import TrainEvalModule

from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd

import torch
import torch.nn as nn

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time
import os
import sys


def main():
    cfg = Config()

    local_time = time.localtime()
    output_dir = f"output/{local_time[0]}_{local_time[1]}_{local_time[2]}_{local_time[3]}_{local_time[4]}_{local_time[5]}"
    os.makedirs(os.path.join(output_dir, "log"))
    os.makedirs(os.path.join(output_dir, "model"))

    pd.DataFrame.from_records([{
        "DATASET_ROOT_PATH": cfg.dataset_root_path,
        "OUPTUT_HM_SHAPE": cfg.output_hm_shape,
        "JOINT_NUM": cfg.joint_num,
        "SIGMA": cfg.sigma,
        "DEVICE": cfg.device,
        "RESNET_TYPE": cfg.resnet_type,
        "BATCH_SIZE": cfg.batch_size,
        "NUM_EPOCHS": cfg.num_epochs,
        "EARLY_STOPPING_PATIENCE": cfg.early_stopping_patience,
        "REDUCE_LR_PATIENCE": cfg.reduce_lr_patience,
        "REDUCE_LR_RATE": cfg.reduce_lr_rate,
        "HEATMAP_LOSS_RATE": cfg.heatmap_loss_rate,
        "CLASS_LOSS_RATE": cfg.class_loss_rate
    }]).to_json(
        f"{os.path.join(output_dir, 'hp.json')}",
        orient="table",
        indent=4,
        index=False
    )

    resize_transform = nn.Sequential(
        transforms.Resize(cfg.output_hm_shape)
    )

    train_df = read_json(f"{os.path.dirname(cfg.dataset_root_path)}/train_dataset_ij.json")
    train_cid = CustomImageDatasetLoadAllIntoMemory(
        cfg=cfg,
        dataset_df=train_df,
        resize_transform=resize_transform
    )
    train_loader = DataLoader(
        train_cid,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    valid_df = read_json(f"{os.path.dirname(cfg.dataset_root_path)}/valid_dataset_ij.json")
    valid_cid = CustomImageDatasetLoadAllIntoMemory(
        cfg=cfg,
        dataset_df=valid_df,
        resize_transform=resize_transform
    )
    valid_loader = DataLoader(
        valid_cid,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    model = Model(cfg=cfg).to(cfg.device)
    summary(
        model=model,
        input_size=(32, 3, 256, 256),
        device=cfg.device
    )
    export_name = os.path.join(output_dir, "model/model(empty)_ij")
    export_model(
        model=model,
        input_size=(32, 3, 256, 256),
        device=cfg.device,
        export_name=export_name
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.lr
    )
    heatmap_loss = nn.MSELoss()
    class_loss = nn.CrossEntropyLoss()
    tem = TrainEvalModule(
        cfg=cfg,
        model=model,
        heatmap_loss=heatmap_loss,
        class_loss=class_loss
    )

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "log"))

    best_valid_total_loss = sys.float_info.max
    early_stopping_cnt = 0
    reduce_lr_cnt = 0

    start_time = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        if epoch % 5 == 0:
            export_name = os.path.join(output_dir, f"model/pt (epoch - {epoch})")
            export_model(
                model=model,
                input_size=(32, 3, 256, 256),
                device=cfg.device,
                export_name=export_name
            )
        model, train_heatmap_loss, train_class_loss, train_class_acc = tem.train(
            train_loader,
            optimizer=optimizer,
            epoch_cnt=epoch,
            class_loss_rate=cfg.class_loss_rate,
            heatmap_loss_rate=cfg.heatmap_loss_rate
        )
        train_total_loss = train_heatmap_loss * cfg.heatmap_loss_rate + train_class_loss * cfg.class_loss_rate
        print(f"\n[EPOCH: {epoch}] - Train Heatmap Loss: {train_heatmap_loss:.4f}; Train Class Loss: {train_class_loss:.4f}; Train Accuracy: {train_class_acc:.2f}%")
        model, valid_heatmap_loss, valid_class_loss, valid_class_acc = tem.evaluate(valid_loader)
        valid_total_loss = valid_heatmap_loss * cfg.heatmap_loss_rate + valid_class_loss * cfg.class_loss_rate
        print(f"\n[EPOCH: {epoch}] - Valid Heatmap Loss: {valid_heatmap_loss:.4f}; Valid Class Loss: {valid_class_loss:.4f}; Valid Accuracy: {valid_class_acc:.2f}%")

        writer.add_scalar("Loss/train_heatmap", train_heatmap_loss, epoch)
        writer.add_scalar("Loss/train_class", train_class_loss, epoch)
        writer.add_scalar("Loss/train_total", train_total_loss, epoch)
        writer.add_scalar("Loss/valid_heatmap", valid_heatmap_loss, epoch)
        writer.add_scalar("Loss/valid_class", valid_class_loss, epoch)
        writer.add_scalar("Loss/valid_total", valid_total_loss, epoch)
        writer.add_scalar("Accuracy/train_class", train_class_acc, epoch)
        writer.add_scalar("Accuracy/valid_class", valid_class_acc, epoch)
        writer.add_scalar("Learning rate/LR", cfg.lr, epoch)

        if valid_total_loss < best_valid_total_loss:
            early_stopping_cnt = 0
            reduce_lr_cnt = 0

            best_valid_total_loss = valid_total_loss

            export_name = os.path.join(output_dir, f"model/pt_[{valid_heatmap_loss:.4f}_{valid_class_loss:.4f}_{valid_total_loss:.4f}]_ij")
            export_model(
                model=model,
                input_size=(32, 3, 256, 256),
                device=cfg.device,
                export_name=export_name,
            )
        else:
            early_stopping_cnt += 1
            reduce_lr_cnt += 1
            print(f"Valid loss didn't improved from {best_valid_total_loss:.4f}; current: {valid_total_loss:.4f}")
            print(f"Early stopping - {early_stopping_cnt} / {cfg.early_stopping_patience}")
            if early_stopping_cnt >= cfg.early_stopping_patience:
                print("Early stopping triggered")
                break
            if reduce_lr_cnt >= cfg.reduce_lr_patience:
                cfg.lr *= cfg.reduce_lr_rate
                reduce_lr_cnt = 0
                print(f"Reduce LR triggered; Current learning rate: {cfg.lr}")
                continue
            print(f"Reduce LR - Current learning rate: {cfg.lr}; {reduce_lr_cnt} / {cfg.reduce_lr_patience}")

    print(f"Training time: {time.time() - start_time}s")


if __name__ == '__main__':
    main()

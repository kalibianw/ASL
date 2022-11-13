from utils import Config, read_json, export_model
from dataset import CustomImageDatasetLoadAllIntoMemory
from model import Model
from trainer import TrainEvalModule

from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time
import os
import sys

RESNET_TYPE = 18
NUM_EPOCHS = 1000
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 3
REDUCE_LR_RATE = 0.5


def main():
    lr = 1e-3

    cfg = Config()
    cfg.resnet_type = RESNET_TYPE
    cfg.batch_size = BATCH_SIZE

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
    export_model(
        model=model,
        input_size=(32, 3, 256, 256),
        device=cfg.device,
        export_name="model/model(empty)_ij"
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    heatmap_loss = nn.MSELoss()
    class_loss = nn.CrossEntropyLoss()
    tem = TrainEvalModule(
        cfg=cfg,
        model=model,
        heatmap_loss=heatmap_loss,
        class_loss=class_loss
    )

    local_time = time.localtime()
    log_path = f"log/{local_time[0]}_{local_time[1]}_{local_time[2]}_{local_time[3]}_{local_time[4]}_{local_time[5]}"
    os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    best_valid_heatmap_loss = sys.float_info.max
    best_valid_class_loss = sys.float_info.max
    early_stopping_cnt = 0
    reduce_lr_cnt = 0

    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch % 5 == 0:
            export_model(
                model=model,
                input_size=(32, 3, 256, 256),
                device=cfg.device,
                export_name=f"model/pt (epoch - {epoch})"
            )
        model, train_heatmap_loss, train_class_loss, train_class_acc = tem.train(train_loader, optimizer=optimizer, epoch_cnt=epoch)
        print(f"\n[EPOCH: {epoch}] - Train Heatmap Loss: {train_heatmap_loss:.4f}; Train Class Loss: {train_class_loss:.4f}; Train Accuracy: {train_class_acc:.2f}%")
        model, valid_heatmap_loss, valid_class_loss, valid_class_acc = tem.evaluate(valid_loader)
        print(f"\n[EPOCH: {epoch}] - Valid Heatmap Loss: {valid_heatmap_loss:.4f}; Valid Class Loss: {valid_class_loss:.4f}; Valid Accuracy: {valid_class_acc:.2f}%")

        writer.add_scalar("Loss/train_heatmap", train_heatmap_loss, epoch)
        writer.add_scalar("Loss/train_class", train_class_loss, epoch)
        writer.add_scalar("Loss/valid_heatmap", valid_heatmap_loss, epoch)
        writer.add_scalar("Loss/valid_class", valid_class_loss, epoch)
        writer.add_scalar("Accuracy/train_class", train_class_acc, epoch)
        writer.add_scalar("Accuracy/valid_class", valid_class_acc, epoch)
        writer.add_scalar("Learning rate/LR", lr, epoch)

        if (valid_heatmap_loss < best_valid_heatmap_loss) & (valid_class_loss < best_valid_class_loss):
            early_stopping_cnt = 0
            reduce_lr_cnt = 0

            best_valid_heatmap_loss = valid_heatmap_loss
            best_valid_class_loss = valid_class_loss

            export_model(
                model=model,
                input_size=(32, 3, 256, 256),
                device=cfg.device,
                export_name=f"model/pt_[{valid_heatmap_loss:.4f}_{valid_class_loss:.4f}]_ij"
            )
        else:
            early_stopping_cnt += 1
            reduce_lr_cnt += 1
            print(f"Valid loss didn't improved from {best_valid_heatmap_loss:.4f}_{best_valid_class_loss:.4f}; current: {valid_heatmap_loss:.4f}_{valid_class_loss:.4f}")
            print(f"Early stopping - {early_stopping_cnt} / {EARLY_STOPPING_PATIENCE}")
            if early_stopping_cnt >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
            if reduce_lr_cnt >= REDUCE_LR_PATIENCE:
                lr *= REDUCE_LR_RATE
                reduce_lr_cnt = 0
                print(f"Reduce LR triggered; Current learning rate: {lr}")
                continue
            print(f"Reduce LR - Current learning rate: {lr}; {reduce_lr_cnt} / {REDUCE_LR_PATIENCE}")

    print(f"Training time: {time.time() - start_time}s")


if __name__ == '__main__':
    main()

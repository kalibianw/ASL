from utils import Config

import torch

from tqdm import tqdm


class TrainEvalModule:
    def __init__(self, cfg: Config, model, heatmap_loss, class_loss):
        self.cfg = cfg

        self.model = model

        self.heatmap_loss = heatmap_loss
        self.class_loss = class_loss

    def train(self, train_loader, optimizer, epoch_cnt):
        self.model.train()
        train_heatmap_loss = 0
        train_class_loss = 0
        train_class_correct = 0

        for batch_idx, (x, y_label, y_lndmrk) in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch_cnt}", total=int(len(train_loader.dataset) / self.cfg.batch_size)):
            x = x.to(self.cfg.device)
            y_label = y_label.to(self.cfg.device).long()
            y_lndmrk = y_lndmrk.to(self.cfg.device)
            batch_heatmaps = list()
            for lndmrk in y_lndmrk:
                batch_heatmaps.append(torch.stack(self.model.render_gaussian_heatmap(lndmrk)))
            y_heatmap = torch.stack(batch_heatmaps)
            y_label = y_label.squeeze(dim=-1)

            optimizer.zero_grad()

            heatmap_out, class_out = self.model(x)

            class_loss_out = self.class_loss(class_out, y_label)
            heatmap_loss_out = self.heatmap_loss(heatmap_out, y_heatmap)

            total_loss = heatmap_loss_out + class_loss_out
            total_loss.backward()

            optimizer.step()

            train_heatmap_loss += heatmap_loss_out.item()
            train_class_loss += class_loss_out.item()
            prediction = class_out.max(1, keepdim=True)[1]
            train_class_correct += prediction.eq(y_label.view_as(prediction)).sum().item()

        train_heatmap_loss /= (len(train_loader.dataset) / self.cfg.batch_size)
        train_class_loss /= (len(train_loader.dataset) / self.cfg.batch_size)
        train_class_acc = 100. * train_class_correct / len(train_loader.dataset)

        return self.model, train_heatmap_loss, train_class_loss, train_class_acc

    def evaluate(self, test_loader):
        self.model.eval()
        test_heatmap_loss = 0
        test_class_loss = 0
        test_class_correct = 0
        with torch.no_grad():
            for batch_idx, (x, y_label, y_lndmrk) in tqdm(test_loader, desc="Evaluate...", total=int(len(test_loader.dataset) / self.cfg.batch_size)):
                x = x.to(self.cfg.device)
                y_label = y_label.to(self.cfg.device).long()
                y_lndmrk = y_lndmrk.to(self.cfg.device)

                y_heatmap = self.model.render_gaussian_heatmap(y_lndmrk)
                y_label = y_label.squeeze(dim=-1)

                heatmap_out, class_out = self.model(x)

                heatmap_loss_out = self.heatmap_loss(heatmap_out, y_heatmap)
                class_loss_out = self.class_loss(class_out, y_label)

                test_heatmap_loss += heatmap_loss_out.item()
                test_class_loss += class_loss_out.item()
                prediction = class_out.max(1, keepdim=True)[1]
                test_class_correct += prediction.eq(y_label.view_as(prediction)).sum().item()

        test_heatmap_loss /= (len(test_loader.dataset) / self.cfg.batch_size)
        test_class_loss /= (len(test_loader.dataset) / self.cfg.batch_size)
        test_class_acc = 100. * test_class_correct / len(test_loader.dataset)

        return self.model, test_heatmap_loss, test_class_loss, test_class_acc

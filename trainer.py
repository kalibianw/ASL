from utils import Config

import torch

import numpy as np

from tqdm import tqdm


class TrainEvalModule:
    def __init__(self, cfg: Config, model, heatmap_loss, class_loss):
        self.cfg = cfg

        self.model = model

        self.heatmap_loss = heatmap_loss
        self.class_loss = class_loss

    def train(self, train_loader, optimizer, epoch_cnt, heatmap_loss_rate=1, class_loss_rate=1):
        self.model.train()
        train_heatmap_loss = 0
        train_class_loss = 0
        train_class_correct = 0

        train_tqdm = tqdm(
            iterable=enumerate(train_loader),
            desc=f"Epoch {epoch_cnt}: heatmap_loss: _; class_loss: _; total_loss: _;",
            total=int(len(train_loader.dataset) / self.cfg.batch_size)
        )
        for batch_idx, (x, y_label, y_lndmrk) in train_tqdm:
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

            total_loss = heatmap_loss_out * heatmap_loss_rate + class_loss_out * class_loss_rate
            total_loss.backward()

            optimizer.step()

            train_heatmap_loss += heatmap_loss_out.item()
            train_class_loss += class_loss_out.item()
            prediction = class_out.max(1, keepdim=True)[1]
            train_class_correct += prediction.eq(y_label.view_as(prediction)).sum().item()

            train_tqdm.set_description(
                desc=f"Epoch {epoch_cnt}: heatmap_loss: {heatmap_loss_out.item():.4f}; class_loss: {class_loss_out.item():.4f}; total_loss: {total_loss.item():.4f};"
            )

        train_heatmap_loss /= (len(train_loader.dataset) / self.cfg.batch_size)
        train_class_loss /= (len(train_loader.dataset) / self.cfg.batch_size)
        train_class_acc = 100. * train_class_correct / len(train_loader.dataset)

        return self.model, train_heatmap_loss, train_class_loss, train_class_acc

    def evaluate(self, test_loader, gaussian_renderer=None, return_cls_output=False):
        self.model.eval()
        test_heatmap_loss = 0
        test_class_loss = 0
        test_class_correct = 0
        cls_target = np.array([])
        cls_output = np.array([])
        with torch.no_grad():
            test_tqdm = tqdm(
                iterable=enumerate(test_loader),
                desc="Evaluate batch_idx _: heatmap_loss: _; class_loss: _; total_loss: _;",
                total=int(len(test_loader.dataset) / self.cfg.batch_size)
            )
            for batch_idx, (x, y_label, y_lndmrk) in test_tqdm:
                x = x.to(self.cfg.device)
                y_label = y_label.to(self.cfg.device).long()
                y_lndmrk = y_lndmrk.to(self.cfg.device)
                batch_heatmaps = list()
                for lndmrk in y_lndmrk:
                    if gaussian_renderer is not None:
                        batch_heatmaps.append(torch.stack(gaussian_renderer.render_gaussian_heatmap(lndmrk)))
                        continue
                    batch_heatmaps.append(torch.stack(self.model.render_gaussian_heatmap(lndmrk)))
                y_heatmap = torch.stack(batch_heatmaps)
                y_label = y_label.squeeze(dim=-1)

                heatmap_out, class_out = self.model(x)

                if return_cls_output:
                    cls_target = np.append(cls_target, y_label.cpu().detach().numpy())
                    cls_output = np.append(cls_output, np.argmax(class_out.cpu().detach().numpy(), axis=1))

                heatmap_loss_out = self.heatmap_loss(heatmap_out, y_heatmap)
                class_loss_out = self.class_loss(class_out, y_label)

                total_loss = heatmap_loss_out + class_loss_out

                test_heatmap_loss += heatmap_loss_out.item()
                test_class_loss += class_loss_out.item()
                prediction = class_out.max(1, keepdim=True)[1]
                test_class_correct += prediction.eq(y_label.view_as(prediction)).sum().item()

                test_tqdm.set_description(
                    f"Evaluate batch_idx {batch_idx}: heatmap_loss: {heatmap_loss_out.item():.4f}; class_loss: {class_loss_out.item():.4f}; total_loss: {total_loss.item():.4f};"
                )

        test_heatmap_loss /= (len(test_loader.dataset) / self.cfg.batch_size)
        test_class_loss /= (len(test_loader.dataset) / self.cfg.batch_size)
        test_class_acc = 100. * test_class_correct / len(test_loader.dataset)

        if return_cls_output is False:
            return self.model, test_heatmap_loss, test_class_loss, test_class_acc
        else:
            return self.model, test_heatmap_loss, test_class_loss, test_class_acc, (cls_target, cls_output)

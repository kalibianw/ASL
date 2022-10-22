from utils import Config
from layer import *

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
from torch.utils import model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type):
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
                       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
                       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
                       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
                       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # RGB
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self):
        org_resnet = model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)

        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")


class PoseNet(nn.Module):
    def __init__(self, cfg: Config):
        super(PoseNet, self).__init__()

        if cfg.resnet_type < 50:
            self.joint_deconv1 = make_deconv_layers([512, 256, 256, 128, 128, 64])
            self.joint_conv1 = make_conv_layers([64, cfg.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

            self.fc = make_linear_layers([512 * 8 * 8, 26], relu_final=False)
        else:
            self.joint_deconv1 = make_deconv_layers([2048, 512, 512, 256, 256, 128])
            self.joint_conv1 = make_conv_layers([128, cfg.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

            self.fc = make_linear_layers([2048 * 8 * 8, 26], relu_final=False)

    def forward(self, img_feat):
        joint_img_feat_1 = self.joint_deconv1(img_feat)
        joint_heatmap_1 = self.joint_conv1(joint_img_feat_1)

        flatten_feat = torch.flatten(img_feat, start_dim=1)
        class_out = nnf.log_softmax(self.fc(flatten_feat), dim=1)

        return joint_heatmap_1, class_out


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super(Model, self).__init__()
        self.cfg = cfg

        self.backbone_net = ResNetBackbone(self.cfg.resnet_type)
        self.posenet = PoseNet(self.cfg)

    def render_gaussian_heatmap(self, joint_coords: torch.Tensor):
        heatmaps = list()
        for joint_coord in joint_coords:
            x = torch.arange(self.cfg.output_hm_shape[1])
            y = torch.arange(self.cfg.output_hm_shape[0])
            yy, xx = torch.meshgrid(y, x)
            xx = xx[:, :].cuda().float()
            yy = yy[:, :].cuda().float()

            x = joint_coord[0].item()
            y = joint_coord[1].item()

            heatmap = torch.exp(-(((xx - x) / self.cfg.sigma) ** 2) / 2 - (((yy - y) / self.cfg.sigma) ** 2) / 2)
            heatmap = heatmap * 255
            heatmaps.append(heatmap)

        return heatmaps

    def forward(self, x):
        img_feat = self.backbone_net(x)
        heatmaps, class_out = self.posenet(img_feat)

        return heatmaps, class_out

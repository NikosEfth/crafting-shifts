import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import timm


class PseudoCombiner(nn.Module):

    def __init__(self, no_classes, pretrained=False, backbone_name="resnet18"):
        super(PseudoCombiner, self).__init__()

        self.backbone_name = backbone_name
        self.backbone, feature_dim = self.create_backbone(
            backbone_name, pretrained, no_classes
        )
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, no_classes)

    def forward(self, x):

        outputs = []
        pseudo_no = len(x)

        # Concat for the backbone forward. This is important for backbones with batch norm.
        x = torch.cat(x, dim=0)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = list(torch.split(x, int(x.shape[0] / pseudo_no), dim=0))

        for idx, pseudo in enumerate(x):
            outputs.append(self.classifier(pseudo))

        if not self.training and len(outputs) > 1:
            outputs.append(
                torch.pow(nn.Softmax(dim=1)(outputs[0]), 0.25)
                * torch.pow(nn.Softmax(dim=1)(outputs[1]), 0.75)
            )
            outputs.append(
                torch.pow(nn.Softmax(dim=1)(outputs[0]), 0.50)
                * torch.pow(nn.Softmax(dim=1)(outputs[1]), 0.50)
            )
            outputs.append(
                torch.pow(nn.Softmax(dim=1)(outputs[0]), 0.75)
                * torch.pow(nn.Softmax(dim=1)(outputs[1]), 0.25)
            )

        return outputs

    def create_backbone(self, backbone_name, pretrained, no_classes):

        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "vit_small":
            backbone = vit_initialization(
                network_variant="vit_small_patch16_224", pretrained=pretrained
            )
            feature_dim = backbone.ft_dim
        elif backbone_name.lower() == "caffenet":
            backbone = AlexNetCaffe()
            for m in backbone.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.0)
            if pretrained:
                state_dict = torch.load("./Pretrained_Models/alexnet_caffe.pth.tar")
                backbone.load_state_dict(state_dict, strict=False)
            backbone.classifier = backbone.classifier[:-1]
            feature_dim = 4096

        return backbone, feature_dim


class vit_initialization(nn.Module):
    def __init__(self, network_variant, pretrained):
        super(vit_initialization, self).__init__()

        self.model = timm.create_model(network_variant, pretrained=pretrained)
        self.ft_dim = self.model.head.in_features

    def forward(self, x):

        x = self.model.forward_features(x)
        x = x[:, 0]  # keep only the cls token as a global descriptor

        return x


class AlexNetCaffe(nn.Module):
    # from https://github.com/fmcarlucci/JigenDG and https://github.com/BUserName/Learning_to_diversify
    def __init__(self, dropout=True):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                    ("norm1", nn.LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                    ("norm2", nn.LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                    ("relu6", nn.ReLU(inplace=True)),
                    ("drop6", nn.Dropout()),
                    ("fc7", nn.Linear(4096, 4096)),
                    ("relu7", nn.ReLU(inplace=True)),
                    ("drop7", nn.Dropout()),
                    ("fc8", nn.Linear(4096, 1000)),
                ]
            )
        )

    def forward(self, x, train=True):
        # 57.6 bring torch data to the range of caffe data
        x = self.features(x * 57.6)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

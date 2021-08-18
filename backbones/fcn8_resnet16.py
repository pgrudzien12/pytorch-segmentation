from collections import OrderedDict

from torch import nn
from typing import Dict

import torch
from torch import Tensor
from torchvision import models

class Resnet4FCN(models.ResNet):
    def _forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        x = self.avgpool(l4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, l1, l2, l3, l4
    def forward(self, x):
        out = OrderedDict()
        # for name, module in self.items():
        #     x = module(x)
        #     if name in self.return_layers:
        #         out_name = self.return_layers[name]
        #         out[out_name] = x
        return out



def fcn_resnet18(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return models.segmentation.segmentation._load_model('fcn', 'resnet18', pretrained, progress, num_classes, aux_loss, **kwargs)

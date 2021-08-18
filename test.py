import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.segmentation.fcn_resnet50(pretrained=True).eval().to(device)
summary(model.classifier, (1, 224, 224))

import torch
import torch.nn as nn
from torchvision.models import resnet50

class HybridModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.cnn = resnet50(pretrained=pretrained)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)



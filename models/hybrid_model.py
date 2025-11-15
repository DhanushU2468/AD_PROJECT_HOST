import torch
import torch.nn as nn
import torchvision.models as models

class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridModel, self).__init__()

        # Load ImageNet pretrained ResNet50
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Convert to accept 1-channel MRI input instead of RGB
        self.base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC layer
        self.base.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.base(x)


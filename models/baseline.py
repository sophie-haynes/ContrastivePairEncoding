import torch.nn as nn
import torchvision.models as models

class BaselineResNet50(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super(BaselineResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)

        if fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


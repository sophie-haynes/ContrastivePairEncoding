import torch
import torch.nn as nn
import torchvision.models as models

class PairDifferenceEncoder(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=128, fine_tune=False):
        super(PairDifferenceEncoder, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        if fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone[-2:].parameters():
                param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, img1, img2, labels=None):
        feat1 = self.backbone(img1).view(img1.size(0), -1)
        feat2 = self.backbone(img2).view(img2.size(0), -1)
        combined_feat = torch.cat((feat1, feat2), dim=1)
        diff_vector = self.fc(combined_feat)

        return {'feats': diff_vector, 'labels': labels}


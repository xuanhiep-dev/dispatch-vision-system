import torch
import torch.nn as nn
from torchvision import models
import timm


class ClsModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ClsModel, self).__init__()

        # Load backbone
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)

        in_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def load(model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device)
        model = ClsModel(num_classes=len(checkpoint['classes']))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint['classes']

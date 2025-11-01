"""ResNet Feature Extractor"""

import os

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_dir: str = "./models/", resnet_variant: str = "resnet50"):
        super(ResNetFeatureExtractor, self).__init__()

        os.makedirs(model_dir, exist_ok=True)
        _model_path = os.path.join(model_dir, f"{resnet_variant}.pth")

        # Validate resnet variant
        if resnet_variant not in ["resnet50", "resnet101", "resnet152"]:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # Load or download model
        if not os.path.exists(_model_path):
            if resnet_variant == "resnet50":
                resnet = resnet50(pretrained=True)
            elif resnet_variant == "resnet101":
                resnet = resnet101(pretrained=True)
            else:  # resnet152
                resnet = resnet152(pretrained=True)
            torch.save(resnet, _model_path)
        else:
            resnet = torch.load(_model_path)

        resnet.eval()

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Set to evaluation mode
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

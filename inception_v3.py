"""Inception V3 Feature Extractor"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import inception_v3


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, model_dir: str = "./models/"):
        super(InceptionV3FeatureExtractor, self).__init__()

        os.makedirs(model_dir, exist_ok=True)
        _model_path = os.path.join(model_dir, "inception_v3.pth")
        if not os.path.exists(_model_path):
            inception = inception_v3(pretrained=True, transform_input=False)
            torch.save(inception, _model_path)
        else:
            inception = torch.load(_model_path)
        inception.eval()

        # Properly extract features by removing auxiliary classifier and final layers
        # We'll manually build the feature extractor without the aux classifier
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        # Set to evaluation mode
        self.eval()

    def forward(self, x):
        # Pass through all layers manually (without aux classifier)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # At this point x should be [batch_size, 2048, 8, 8]
        # Apply global average pooling to get [batch_size, 2048, 1, 1]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten to [batch_size, 2048]
        x = x.view(x.size(0), -1)
        return x

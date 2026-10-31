import torch
import torch.nn as nn
from torchvision import models


class ODIRclassifier(nn.Module):
    def __init__(self, n_classes=8, attr_dim=2):
        super().__init__()

        self.left_image_net = models.efficientnet_b0(weights="DEFAULT")
        self.right_image_net = models.efficientnet_b0(weights="DEFAULT")

        f = self.left_image_net.classifier[1].in_features  # 1280

        self.left_image_net.classifier = nn.Identity()
        self.right_image_net.classifier = nn.Identity()

        self.attributes_vec_fc = nn.Sequential(
            nn.Linear(attr_dim, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(f * 2 + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes)
        )

    def forward(self, left_image, right_image, attributes_vec):
        left_features = self.left_image_net(left_image)
        right_features = self.right_image_net(right_image)
        attr_features = self.attributes_vec_fc(attributes_vec)

        combined = torch.cat([left_features, right_features, attr_features], dim=1)
        return self.classifier(combined)
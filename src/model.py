import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Custom CNN for three-class long bone fracture severity classification.

    Classes:
        0 - Healthy
        1 - Simple
        2 - Wedge_Complex
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 224x224 -> 14x14 after four 2x2 pooling operations
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
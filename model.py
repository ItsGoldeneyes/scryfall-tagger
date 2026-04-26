from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LightingClassifier(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, 2),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout inference.

        Runs n_passes forward passes with dropout active, then returns
        (mean_probs, uncertainty) where uncertainty is the mean predictive
        variance across classes — higher means the model is less sure.
        """
        self.train()  # activates dropout
        with torch.no_grad():
            probs = torch.stack(
                [torch.softmax(self(x), dim=1) for _ in range(n_passes)]
            )
        self.eval()
        mean_probs = probs.mean(0)
        # sum variance across class dimension so uncertainty is a scalar per image
        uncertainty = probs.var(0).sum(1)
        return mean_probs, uncertainty

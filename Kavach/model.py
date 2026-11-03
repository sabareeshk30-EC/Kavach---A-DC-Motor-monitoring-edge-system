import torch
import torch.nn as nn


class MotorConditionCNN(nn.Module):
    """
    MotorConditionCNN v2 — improved architecture for Kavach EdgeInspect.

    Input shape: [B, 1, 64, T]  (log-mel spectrogram, single channel)

    Architecture
    ------------
    Stage 1: Conv(1->32)  + BN + ReLU + MaxPool(2)   -> [B, 32, 32, T/2]
    Stage 2: Conv(32->64) + BN + ReLU + MaxPool(2)   -> [B, 64, 16, T/4]
    Stage 3: Conv(64->128)+ BN + ReLU + MaxPool(2)   -> [B, 128, 8, T/8]
    Stage 4: Conv(128->128)+BN + ReLU + AdaptAvgPool -> [B, 128, 4, 4]
    Flatten -> Dropout(0.4) -> FC(2048->256) -> ReLU
            -> Dropout(0.3) -> FC(256->num_classes)

    BatchNorm after every Conv layer speeds convergence and regularises
    the small acoustic dataset. Dropout before both FC layers prevents
    co-adaptation of features.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        def _block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stage1 = nn.Sequential(_block(1, 32),   nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(_block(32, 64),  nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(_block(64, 128), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(
            _block(128, 128),
            nn.AdaptiveAvgPool2d((4, 4)),   # fixed spatial size regardless of T
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x


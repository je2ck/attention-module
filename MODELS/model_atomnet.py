import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AtomCBAMNet(nn.Module):
    """
    입력  : (B, 2, 16, 16)  # [raw_normalized, denoised]
    출력  : (B, num_classes)
    구조  :
      [Stage1] 2→32 conv×2 + CBAM(32, r=r1) + MaxPool(2)    # 16→8
      [Stage2] 32→64 conv×2 + CBAM(64, r=r2) + MaxPool(2)   # 8→4
      GAP → Linear(64→num_classes)

    작은 ROI에 맞춰 과도한 다운샘플/거대한 FC를 피하고,
    각 stage에서 채널/공간 주의를 재보정합니다.
    """
    def __init__(self, num_classes: int,
                 in_channels: int = 2,
                 r1: int = 8,
                 r2: int = 8,
                 dropout: float = 0.0):
        super().__init__()

        # Stage 1 (16x16 -> 8x8)
        self.s1a = conv_bn_relu(in_channels, 32, k=3, s=1, p=1)
        self.s1b = conv_bn_relu(32, 32, k=3, s=1, p=1)
        self.cbam1 = CBAM(32, r1)          # 네 cbam.py의 인터페이스 그대로
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2 (8x8 -> 4x4)
        self.s2a = conv_bn_relu(32, 64, k=3, s=1, p=1)
        self.s2b = conv_bn_relu(64, 64, k=3, s=1, p=1)
        self.cbam2 = CBAM(64, r2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B,64,1,1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(64, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,2,16,16)
        x = self.s1a(x)
        x = self.s1b(x)
        x = self.cbam1(x)
        x = self.pool1(x)          # (B,32,8,8)

        x = self.s2a(x)
        x = self.s2b(x)
        x = self.cbam2(x)
        x = self.pool2(x)          # (B,64,4,4)

        x = self.gap(x)            # (B,64,1,1)
        x = x.flatten(1)           # (B,64)
        x = self.drop(x)
        x = self.fc(x)             # (B,num_classes)
        return x


# 사용 예시
if __name__ == "__main__":
    model = AtomCBAMNet(num_classes=2, in_channels=2, r1=8, r2=8, dropout=0.1)
    dummy = torch.randn(8, 2, 16, 16)
    out = model(dummy)
    print(out.shape)  # torch.Size([8, 2])
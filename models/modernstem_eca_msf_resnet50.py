import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck


class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y


class ECABottleneck(Bottleneck):
    def __init__(self, *args, use_eca=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.eca = ECAModule(self.bn3.num_features) if use_eca else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleFusion(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        self.proj3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.proj5 = nn.Conv2d(2048, out_channels, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, c3, c4, c5):
        p3 = self.proj3(c3)

        p4 = self.proj4(c4)
        p4 = F.interpolate(
            p4,
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        p5 = self.proj5(c5)
        p5 = F.interpolate(
            p5,
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        fused = torch.cat([p3, p4, p5], dim=1)
        fused = self.fuse(fused)

        return fused


class ModernECAMSFResNet50(ResNet):
    def __init__(self, num_classes=23, dropout=0.3):
        super().__init__(
            block=ECABottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            zero_init_residual=True
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.msf = MultiScaleFusion(out_channels=256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_head = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self._init_new_layers()

    def _init_new_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        x = self.msf(c3, c4, c5)

        avg = self.avgpool(x).flatten(1)
        mx = self.maxpool_head(x).flatten(1)

        x = torch.cat([avg, mx], dim=1)
        x = self.fc(x)

        return x


def build_model(num_classes=23):
    return ModernECAMSFResNet50(num_classes=num_classes)
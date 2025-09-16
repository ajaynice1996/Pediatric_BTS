import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Double Conv Block
# -----------------------------
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# -----------------------------
# Downsample Block
# -----------------------------
class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

# -----------------------------
# Upsample Block
# -----------------------------
class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_ch, out_ch, in_ch//2)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding in case input sizes differ
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW//2, diffW - diffW//2,
                        diffH//2, diffH - diffH//2,
                        diffD//2, diffD - diffD//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -----------------------------
# Output Block
# -----------------------------
class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# -----------------------------
# UNet3D
# -----------------------------
class UNet3D(nn.Module):
    print(" I am inside UNet3D...............")
    def __init__(self, in_channels=4, n_classes=3, base_channels=16):
        super().__init__()
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels*2)
        self.down2 = Down3D(base_channels*2, base_channels*4)
        self.down3 = Down3D(base_channels*4, base_channels*8)
        self.bottleneck = DoubleConv3D(base_channels*8, base_channels*16)
        self.up3 = Up3D(base_channels*16 + base_channels*8, base_channels*8)
        self.up2 = Up3D(base_channels*8 + base_channels*4, base_channels*4)
        self.up1 = Up3D(base_channels*4 + base_channels*2, base_channels*2)
        self.up0 = Up3D(base_channels*2 + base_channels, base_channels)
        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.bottleneck(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        x = self.outc(x)
        return x

# -----------------------------
# Testing
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(in_channels=4, n_classes=4).to(device)
    x = torch.randn(1, 4, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(out.shape)  # expected: (1,3,96,128,128)

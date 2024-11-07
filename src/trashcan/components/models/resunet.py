from torch import nn
import torch

from trashcan_core.components.models import Net
from trashcan_core.components.constants import N_CLASSES


class ResUNet(Net):
    def __init__(self):
        super(ResUNet, self).__init__()

        self.encoder1 = self._conv_block(3, 32)
        self.encoder2 = self._conv_block(32, 64)
        self.encoder3 = self._conv_block(64, 128)
        self.encoder4 = self._conv_block(128, 256)
        self.encoder5 = self._conv_block(256, 512)

        self.bottleneck = self._conv_block(512, 1024)

        self.upconv5 = self._upconv_block(1024, 512)
        self.upconv4 = self._upconv_block(512, 256)
        self.upconv3 = self._upconv_block(256, 128)
        self.upconv2 = self._upconv_block(128, 64)
        self.upconv1 = self._upconv_block(64, 32)

        self.final_conv = nn.Conv2d(32, N_CLASSES, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        enc5 = self.encoder5(nn.MaxPool2d(2)(enc4))

        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self._conv_block(1024, 512)(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self._conv_block(512, 256)(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self._conv_block(256, 128)(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self._conv_block(128, 64)(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self._conv_block(64, 32)(dec1)

        out = self.final_conv(dec1)
        return out

from torch import nn

from trashcan.components.models import ResUNet
from trashcan_core.components.constants import N_CLASSES


class ResUNetMini(ResUNet):
    def __init__(self):
        super(ResUNetMini, self).__init__()

        self.encoder1 = self._conv_block(3, 8)
        self.encoder2 = self._conv_block(8, 16)
        self.encoder3 = self._conv_block(16, 32)
        self.encoder4 = self._conv_block(32, 64)
        self.encoder5 = self._conv_block(64, 128)

        self.bottleneck = self._conv_block(128, 256)

        self.upconv5 = self._upconv_block(256, 128)
        self.upconv4 = self._upconv_block(128, 64)
        self.upconv3 = self._upconv_block(64, 32)
        self.upconv2 = self._upconv_block(32, 16)
        self.upconv1 = self._upconv_block(16, 8)

        self.decoder1 = self._conv_block(16, 8)
        self.decoder2 = self._conv_block(32, 16)
        self.decoder3 = self._conv_block(64, 32)
        self.decoder4 = self._conv_block(128, 64)
        self.decoder5 = self._conv_block(256, 128)

        self.final_conv = nn.Conv2d(8, N_CLASSES, kernel_size=1)

        # self.decoder1 = self._conv_block(8, 4)
        # self.decoder2 = self._conv_block(16, 8)
        # self.decoder3 = self._conv_block(32, 16)
        # self.decoder4 = self._conv_block(64, 32)
        # self.decoder5 = self._conv_block(128, 64)

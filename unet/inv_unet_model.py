""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .inv_unet_parts import *


class InvNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # n_channels input channels, grey is 1, rgb is 3, n_classes also
        super(InvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(n_channels, 256)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)

        self.up1 = Up(512, 512, 0.5)  # dropout rate 0.5, 
        self.up2 = Up(1024, 512, 0.5)  # dropout rate 0.5
        self.up3 = Up(1024, 512, 0.5)  # dropout rate 0.5
        self.up4 = Up(768, 256)
        self.up5 = Up(512, 256)
        self.up6 = Up(512, 256)
        self.conv1 = TripleConv(n_channels+256, 128, 64, 32)  # 257 + 256 is related to image size, 288 for local test
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        #x1 = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x7 = self.up1(x6, x5)
        x7 = self.up2(x7, x4)
        x7 = self.up3(x7, x3)
        x7 = self.up4(x7, x2)
        x7 = self.up5(x7, x1)
        x =  self.up6(x7, x)

        x = self.conv1(x)
        output = self.outc(x)
        return output

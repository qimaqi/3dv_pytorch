""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .inv_unet_parts import *


class InvNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # n_channels input channels, grey is 1, rgb is 3, n_classes also
        super(InvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #self.bilinear = bilinear

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
        self.conv1 = TripleConv(n_channels + 256, 128, 64, 32)  # 256 + 256 is related to image size, 288 for local test
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # print('x',x,'  x size', x.size())  #[1,32,168,224]
        x1 = self.down1(x)
        #print(' x1 size', x1.size())  #([1, 256, 84, 112])
        x2 = self.down2(x1)
        #print(' x2 size', x2.size()) # ([1, 256, 42, 56])
        x3 = self.down3(x2)
        #print(' x3 size', x3.size())  # ([1, 256, 21, 28])
        x4 = self.down4(x3)
        # print(' x4 size', x4.size())  # ([1, 512, 10, 14])
        x5 = self.down5(x4)
        # print(' x5 size', x5.size())  # ([1, 512, 5, 7])
        x6 = self.down6(x5)
        # print(' x6 size', x6.size()) # ([1, 512, 2, 3])

        x7 = self.up1(x6, x5)
        #print(' x7 size', x7.size()) # ([1, 1024, 5, 7])
        x8 = self.up2(x7, x4)
        #print(' x8 size', x8.size()) # ([1, 1024, 10, 14]) # x7 conv to 512 channel
        x9 = self.up3(x8, x3)
        #print(' x9 size', x9.size()) # ([1, 768, 21, 28]) 
        x10 = self.up4(x9, x2)
        # print(' x10 size', x10.size()) # ([1, 512, 42, 56])
        x11 = self.up5(x10, x1)
        # print(' x11 size', x11.size()) # ([1, 512, 84, 112])
        x12 =  self.up6(x11, x)
        # print(' x12 size', x12.size()) # [1, 288, 168, 224])
        x13 = self.conv1(x12)
        # print(' x13 size', x13.size()) # ([1, 32, 168, 224])

        output = self.outc(x13)
        #print(' output size', output.size()) # ([1, 1, 168, 224])
        return output

class VisbNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # n_channels input channels, grey is 1, rgb is 3, n_classes also
        super(InvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 64)
        if n_channels < 5:
            self.down1 = Down(n_channels, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
        else:
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
        if n_channels < 5:
            self.up5 = Up(384, 256)
            self.up6 = Up(320, 256)
        else:
            self.up5 = Up(512, 256)
            self.up6 = Up(512 ,256)
        self.conv1 = TripleConv(256 + n_channels, 128, 64, 32)  # 256 + 256 is related to image size, 288 for local test
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

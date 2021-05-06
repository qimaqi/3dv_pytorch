""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if n_channels == 256:    # Superpoint
            self.inc = DoubleConv(n_channels, 256)   # replace 256 with 128 for R2D2
            self.down1 = Down(256, 256)
            self.down2 = Down(256, 256)  # keep 256 for superpoint
            self.down3 = Down(256, 512)  # do not higher
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 512)
            self.up1 = Up(1024, 256, bilinear) # 256
            self.up2 = Up(512, 256, bilinear) # 256
            self.up3 = Up(512, 256, bilinear) # 256
            self.up4 = Up(512, 256, bilinear) # 256
            self.outc = OutConv(256, n_classes)

        elif n_channels == 128:  # R2D2
            self.inc = DoubleConv(n_channels, 128)   # replace 256 with 128 for R2D2
            self.down1 = Down(128, 128)
            self.down2 = Down(128, 128)  
            self.down3 = Down(128, 256)  # do not higher than 512 channel
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 256)
            self.up1 = Up(512, 128, bilinear) 
            self.up2 = Up(256, 128, bilinear) 
            self.up3 = Up(256, 128, bilinear) 
            self.up4 = Up(256, 128, bilinear) 
            self.outc = OutConv(128, n_classes)
        
        elif n_channels == 257:    # Superpoint
            self.inc = DoubleConv(n_channels, 257)   # replace 256 with 128 for R2D2
            self.down1 = Down(257, 257)
            self.down2 = Down(257, 257)  # keep 256 for superpoint
            self.down3 = Down(257, 512)  # do not higher
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 512)
            self.up1 = Up(1024, 256, bilinear) # 256
            self.up2 = Up(513, 256, bilinear) # 256
            self.up3 = Up(513, 256, bilinear) # 256
            self.up4 = Up(513, 256, bilinear) # 256
            self.outc = OutConv(256, n_classes)

        elif n_channels == 129:  # R2D2
            self.inc = DoubleConv(n_channels, 129)   # replace 256 with 128 for R2D2
            self.down1 = Down(129, 129)
            self.down2 = Down(129, 129)  
            self.down3 = Down(129, 256)  # do not higher than 512 channel
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 256)
            self.up1 = Up(512, 128, bilinear) 
            self.up2 = Up(257, 128, bilinear) 
            self.up3 = Up(257, 128, bilinear) 
            self.up4 = Up(257, 128, bilinear) 
            self.outc = OutConv(128, n_classes)
        


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # x5:256, x4:256, output 128
        x = self.up2(x, x3)  # x:128, x3:128, output 128
        x = self.up3(x, x2)  # x:128 x2:128 output 128
        x = self.up4(x, x1)  # x:128, x1:128, output 128
        logits = self.outc(x) # x 128, output n_classes
        return logits
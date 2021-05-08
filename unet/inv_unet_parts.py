""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, mid_channel_1, mid_channel_2, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, mid_channel_1, kernel_size=3, stride = 1),
            nn.BatchNorm2d(mid_channel_1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(mid_channel_1, mid_channel_2, kernel_size=3, stride=1),
            nn.BatchNorm2d(mid_channel_2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(mid_channel_2, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop_rate = 0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),  # maybe bilinear and
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(p=drop_rate, inplace=False),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # first cat or conv???

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.Tanh() #nn.ReLU(inplace=True)  # nn.Tanh()
        )
    def forward(self, x):
        return self.outconv(x)


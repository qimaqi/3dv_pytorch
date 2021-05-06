# edit by Qi Ma
# qimaqi@student.ethz.ch

import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class VGGPerception(nn.Module):
    def __init__(self):
        super(VGGPerception, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.mean = torch.nn.Parameter(torch.tensor([0.485*255, 0.456*255, 0.406*255], device = device, requires_grad = False).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229*255, 0.224*255, 0.225*255], device = device, requires_grad = False).view(1,3,1,1))
        #self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device=device, dtype=torch.float32)
        #self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device=device, dtype=torch.float32)

        for x in range(4):  # conv1_1, relu, conv1_2, relu  4
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):  # max_pool, conv2_1, relu, conv2_2, relu   5
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16): # maxpool, conv3_1, relu, conv3_2, relu, conv3_3, relu 7 
            self.to_relu_3_3.add_module(str(x), features[x])
        #for x in range(16, 23):
        #    self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        

    def forward(self, x):  # input x:HxW, target y
        # x is a reconstructed image, y is a ground truth image, x y size same
        # input just need to be channel 3
        #if input.shape[1] != 3:  the 3 is the RGB channel
        #input = input.repeat(1, 3, 1, 1)
        #target = target.repeat(1, 3, 1, 1)   we have to transpose to tensor
        if x.shape[1] != 3:  # grey image
            #x = np.expand_dims(x, axis=2)
            tx = x.repeat(1, 3, 1, 1)
        else:
            tx = x
        #zero_channel_x = np.zeros(x.shape)
        #x_rgb = np.concatenate((x,zero_channel_x,zero_channel_x),axis=2)
        #x_trans = x_rgb.transpose((2, 0, 1))  # CHW
        #tx = torch.from_numpy(x_trans).type(torch.FloatTensor)

        tx = (tx-self.mean) / self.std

        px = self.to_relu_1_2(tx)
        p_relu_1_2 = px
        px = self.to_relu_2_2(px)
        p_relu_2_2 = px
        px = self.to_relu_3_3(px)
        p_relu_3_3 = px

        out = (p_relu_1_2, p_relu_2_2, p_relu_3_3)
        return out




class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    #if pretrained:
    #    state_dict = load_state_dict_from_url(model_urls[arch],
    #                                          progress=progress)
    #    model.load_state_dict(state_dict)
    model.load_state_dict(torch.load('./vgg16-397923af.pth'))
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
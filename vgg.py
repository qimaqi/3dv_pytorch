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
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device = device, requires_grad = False).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device = device, requires_grad = False).view(1,3,1,1))
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
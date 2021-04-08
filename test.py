import argparse
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.dataset import BasicDataset2
from eval import eval_net
from unet import InvNet
#from unet import LossNetwork

from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

imgs_dir = 'data/nyu_v1_images/'
dir_features = 'data/nyu_v1_features/'
ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
img_file = glob(imgs_dir + ids[0] + '.*')
feature_file = glob(dir_features + ids[0] + '.*')
#img = Image.open(img_file[0]).convert('L')
img = cv2.imread(img_file[0],0)
feature = np.load(feature_file[0])['feature']
# print(img.size)
# print(np.shape(img))
# print(np.shape(feature)[:2])
# HWC to CHW ... in our feature which is 640x480 WHC -> CHW
img_nd = feature
img_trans = img_nd.transpose((2, 1, 0))
img_trans = (img_trans/127.5)-1
t1 = torch.from_numpy(img).type(torch.FloatTensor)
t2 = torch.from_numpy(img_trans).type(torch.FloatTensor)
#print(img_trans.shape)
#print(t2.shape[1])
pixel_loss = nn.L1Loss()
loss = pixel_loss(cpred, true_imgs)
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

    

if __name__ == '__main__':
    val_percent = 0.1
    #dataset = BasicDataset2(imgs_dir)
    #print((type(dataset)))
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = random_split(dataset, [n_train, n_val])
    #print(type(train))
    #print(type(val))
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #eval_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    #criterion = VGGPerceptualLoss()
    #target = img
    #inp = img
    #print(inp.shape) 
    #loss = criterion(inp,target)

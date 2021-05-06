import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.dataset import BasicDataset3
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
import json
from torchvision.utils import save_image


infer_output_dir = 'F:/invsfm/src/data/infer_output/'
dir_desc = 'F:/invsfm/src/data/infer_desc/'
dir_checkpoint = '/cluster/scratch/qimaqi/checkpoints_11_4/5.pth'
dir_depth = 'F:/invsfm/src/data/infer_depth/'
dir_pos = 'F:/invsfm/src/data/infer_pos/'
dir_img = 'F:/invsfm/src/data/infer_imgs/' 

img_scale = 0.8
pct_3D_points = 0
crop_size = 256
batch_size = 1
dataset = BasicDataset3(dir_img, dir_depth, dir_pos, dir_desc, img_scale, pct_3D_points, crop_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_train = len(dataset)

def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename, normalize=True)

global_step = 0
for batch in train_loader:
    input_features = batch['feature']
    true_imgs = batch['image']
    #assert input_features.shape[1] == net.n_channels, 'Channel match problem'

    input_features = input_features.to(device=device, dtype=torch.float32)
    #mask_type = torch.float32
    true_imgs = true_imgs.to(device=device, dtype=torch.float32)
    #input_tensor = input_tensor.clone().detach()
    #input_tensor = input_tensor.to(torch.device('cpu'))



    global_step += 1
    # debug part
    if global_step % (n_train // (10 * batch_size)) == 0:
        tmp_output_dir = 'F:/invsfm/src/data/debug_output/' +str(global_step) + '.png'
        tmp_img_dir = 'F:/invsfm/src/data/debug_images/'+ str(global_step) + '.png'
        #save_image_tensor(cpred,tmp_output_dir)
        save_image_tensor(true_imgs,tmp_img_dir)
        #print('cpred maximum', torch.max(cpred))
        #print('cpred minimum', torch.min(cpred))
        print('true_images maximum', torch.max(true_imgs))
        print('true_images minimum', torch.min(true_imgs))

# imgs_dir = '../data/nyu_v1_images/'
# dir_features = '../data/nyu_v1_features/'
# ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
# img_file = glob(imgs_dir + ids[0] + '.*')
# feature_file = glob(dir_features + ids[0] + '.*')
# #print(feature_file)
# f1 = np.arange(100)
# print(f1)
# print(f1.shape)
# f1.resize(5,20)
# print(f1)
# f1.resize(20,5)
# print(f1)


#img = Image.open(img_file[0]).convert('L')
#img = cv2.imread(img_file[0],0)
#img2 = cv2.imread(img_file[0],0)
#feature = np.load(feature_file[0])['feature']
# print(img.size)
# print(np.shape(img))
#print(np.shape(feature))
#print(feature[:40,:40,0])
#f1 = {}
#f1['feature1'] = feature[:32,:24,0].tolist()
#w = 320
#h = 240
#feature =  np.resize(feature,(w,h,256))
#f1['feature2'] = feature[:32,:24,0].tolist()
#print(feature[:40,:40,0])
# HWC to CHW ... in our feature which is 640x480 WHC -> CHW
#path = './resizeshow.json'
#with open(path,'w') as desc_dict_file:
    #desc_dict_file.write(desc_dict_w)
#    json.dump(f1, desc_dict_file)
#img_nd = feature
#img_trans = img_nd.transpose((2, 1, 0))
#img_trans = (img_trans/127.5)-1
#t1 = torch.from_numpy(img).type(torch.FloatTensor)
#t2 = torch.from_numpy(img2).type(torch.FloatTensor)
#print(img_trans.shape)
#print(t2.shape[1])
#pixel_loss = nn.L1Loss()
#loss = pixel_loss(cpred, true_imgs)

#print(grey.shape)
#input = grey.repeat(1, 3, 1, 1)
#img = np.expand_dims(img, axis=2)
#zero_channel_1 = np.zeros(img.shape)
#zero_channel_2 = np.zeros(img.shape)
#rgb_img = np.concatenate((img,zero_channel_1,zero_channel_2),axis=2)
#print(rgb_img.shape)
#img_trans = rgb_img.transpose((2, 0, 1))  # CHW
#t0 = torch.from_numpy(img).type(torch.FloatTensor)
#t1 = torch.from_numpy(zero_channel_1).type(torch.FloatTensor)
#t2 = torch.from_numpy(zero_channel_2).type(torch.FloatTensor)
#t_3 = torch.cat((t0,t1,t2),0)

#print(t0.shape)


#from vgg import VGGPerception
#criterion = VGGPerception()
#l2_loss = nn.MSELoss()
#loss1 = criterion(img)
#loss2 = criterion(img2)
#loss3 = l2_loss(loss1[0],loss2[0])
#print(loss3)




#print(input.shape)


#import torchvision

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

    

#if __name__ == '__main__':
    #val_percent = 0.1
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
    #path = 'data/nyu_v1_images/0.jpg'
    #grey = cv2.imread(path,0)
    #grey = torch.from_numpy(grey).type(torch.FloatTensor)
    #print(grey)
    #input = grey.repeat(1, 3, 1, 1)
    #print(input)

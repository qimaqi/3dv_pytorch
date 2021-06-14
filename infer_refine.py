import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from vgg import VGGPerception
from utils.dataset import dataset_superpoint_5k
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from unet import UNet
from PIL import Image
import os
import numpy as np

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

infer_output_dir = './refine_out/'
dir_checkpoint = './checkpoints/11.pth'
base_image_dir = '/home/wangr/invsfm/data'
base_feature_dir = '/home/wangr/superpoint_resize/resize_data_superpoint_1'
train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
train_5k_image_rgb=list(train_5k[:,4])

image_list=[]

feature_list=[]
for i in range(len(train_5k_image_rgb)):
    temp_image_name=train_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    image_list.append(temp_path)
    superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))

    


def run_infer(net,refine,infer_loader,device):
  count = 1
  for batch in infer_loader:
    try:
        with torch.no_grad():
            net.to(device=device)
            net.eval()
            refine.to(device=device)
            refine.eval()
            input_features = batch['feature']
            name_i = str(count)#batch['name']
            print(name_i)
            # name_i = name_i[0]
            # name_i = name_i.replace('/','^_^')
            input_features = input_features.to(device=device, dtype=torch.float32)
            pred = net(input_features).detach()
            refine_input = torch.cat((pred,input_features),axis=1)
            pred = refine(refine_input)
            pred = (pred+1.)*127.5
            pred = pred.to('cpu')
            ouput_path = infer_output_dir + name_i+ '.png'
            save_image_tensor(pred,ouput_path)
            del pred 
            torch.cuda.empty_cache()
            count = count +1
    except:
         with torch.no_grad():
            net.to(device='cpu')
            net.eval()
            refine.to(device='cpu')
            refine.eval()
            input_features = batch['feature']
            name_i = str(count) #batch['name']
            print(name_i)
            # name_i = name_i[0]
            # name_i = name_i.replace('/','^_^')
            input_features = input_features.to(device='cpu', dtype=torch.float32)
            pred = net(input_features).detach()
            refine_input = torch.cat((pred,input_features),axis =1 )
            pred =  refine(refine_input)
            pred = (pred+1.)*127.5
            pred = pred.to('cpu')
            ouput_path = infer_output_dir + name_i+ '.png'
            save_image_tensor(pred,ouput_path)
            del pred 
            torch.cuda.empty_cache()
            count = count+1

def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    save_image(input_tensor, filename, normalize=True)


if __name__ == '__main__':
    batch_size = 1
    img_scale = 1
    crop_size = 0
    pct_3D_points = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=256, n_classes=1)   # input should be 256, resize to 32 so ram enough
    net.load_state_dict(
        torch.load('./coarse.pth')
        )
    refine = UNet(n_channels = 257, n_classes = 3)

    refine.load_state_dict(torch.load('./refine_checkpoint/5.pth'))
    dataset = dataset_superpoint_5k(image_list,feature_list,img_scale, pct_3D_points, crop_size)
    infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    n_infer = int(len(dataset))
    run_infer(net,refine,infer_loader,device)

        

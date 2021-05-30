import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from vgg import VGGPerception
from utils.dataset import dataset_superpoint_5k, dataset_superpoint_5k_online_infer
from torch.utils.data import DataLoader

from torchvision.utils import save_image
#from unet import UNet
from unet import UNet_Nested
from PIL import Image
import os
import numpy as np

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# infer_output_dir = './infer_256/'
# dir_checkpoint = './checkpoints/11.pth'
# base_image_dir = '/home/wangr/invsfm/data'
# base_feature_dir = '/home/wangr/superpoint_resize/resize_data_superpoint_1'

# train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
# train_5k_image_rgb=list(train_5k[:,4])
infer_output_dir ='/cluster/scratch/qimaqi/data_5k/val_unet++_1000_cpu_origin_max1000_9/' #'/cluster/scratch/qimaqi/data_5k/test_unet_6000_cpu_origin_max20000/'
dir_checkpoint = '/cluster/scratch/qimaqi/checkpoints_27_5_unet++_1000/4.pth' # '/cluster/scratch/qimaqi/checkpoints_28_5_unet_max_6000_lr1e-4/4.pth' # '/cluster/scratch/qimaqi/checkpoints_27_unet_online_max_1000_lr1e-4/8.pth' #'/cluster/scratch/qimaqi/checkpoints_28_unet_online_max_2000_lr1e-4/7.pth' 
base_image_dir = '/cluster/scratch/qimaqi/data_5k/data' 
base_feature_dir  = '/cluster/scratch/qimaqi/data_5k/save_source_dir/resize_data_superpoint_1'
# unet++
#'/cluster/scratch/qimaqi/checkpoints_27_5_unet++_6000/4.pth'
#

try:
    os.mkdir(infer_output_dir)
    logging.info('Created infer_output_dir directory')
except OSError:
    pass


test_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/test.txt'))
test_5k_image_rgb=list(test_5k[:,4])
val_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
val_5k_image_rgb=list(val_5k[:,4])


test_list = [32,36,60,79,116,118, 317, 335, 476, 499, 533, 587, 610, 650, 1003]
image_list=[]
feature_list=[]
for i in (32,36,60,79,116,118, 317, 335, 476, 499, 533, 587, 610, 650, 1003): #range(len(test_5k_image_rgb)):  # problem have in 32: 753x502, 36:960x707, 60: 795x1200, 79: 797x1200, 116: 1200x890, 118:1200x893, 317: 1200x786, 335:681x1200 , 476: 1200x789, 499:1200x609  
    temp_image_name=test_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    image_list.append(temp_path)
    superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))


val_list = [7,11,53,56,103,113,149,173,200,217,267,306,309,425,431,535,537,564,577,621,657,984]
val_image_list=[]
val_feature_list=[]
for i in (564,577): #,11,53,56,103,113,149,173,200,217,267,306,309,425,431,535,537,564,577,621,657,984): #range(len(val_5k_image_rgb)):  # problem have in 7: 636x960, 53:1200x855,  56: 896x584, 103:1200x900  113: 757x631  149: 888x1200
    temp_image_name=val_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    val_image_list.append(temp_path)
    superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    val_feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))



def run_infer(net,infer_loader,device):
    i = 0
    for batch in infer_loader:
        try:
            with torch.no_grad():
                # if True: 
                #if i in (11,56,103,149,217,425,309,7,53,113,173,200,267,306,431,535,537,564,577,621,657,984):
                # if i in (32, 317, 335, 476, 499, 36, 60, 79, 116, 118, 533, 587, 610, 650, 1003)
                net.to(device=device)
                net.eval()
                input_features = batch['feature']
                #name_i = batch['name']
                #print(name_i)
                #name_i = name_i[0]
                #name_i = name_i.replace('/','^_^')
                input_features = input_features.to(device=device, dtype=torch.float32)
                pred = net(input_features).detach()
                pred = (pred+1.)*127.5
                pred = pred.to('cpu')
                ouput_path = infer_output_dir + str(i)+ '.png'
                save_image_tensor(pred,ouput_path)
                # print('finish already ',test_list[i])
                print('finish already ',i)#val_list[i])
                del pred 
                torch.cuda.empty_cache()
                
        except:
            print('image have problem in ', i)
            pass
        i+=1
            # with torch.no_grad():
            #     net.to(device='cpu')
            #     net.eval()
            #     input_features = batch['feature']
            #     #name_i = batch['name']
            #     #print(name_i)
            #     #name_i = name_i[0]
            #     name_i = name_i.replace('/','^_^')
            #     input_features = input_features.to(device='cpu', dtype=torch.float32)
            #     pred = net(input_features).detach()
            #     pred = (pred+1.)*127.5
            #     pred = pred.to('cpu')
            #     ouput_path = infer_output_dir + str(i)+ '.png'
            #     save_image_tensor(pred,ouput_path)
            #     del pred 
            #     torch.cuda.empty_cache()
            #     i+=1
        # print('finish already ',i)
        # if i == 100:
        #    break

def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    # to cpu
    # input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename, normalize=True)


if __name__ == '__main__':
    batch_size = 1
    img_scale = 1
    crop_size = 0
    pct_3D_points = 0
    max_points = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #torch.device('cpu')#

    # net = UNet(n_channels=256, n_classes=1)   # input should be 256, resize to 32 so ram enough
    net = UNet_Nested(n_channels=256, n_classes=1)
    net.load_state_dict(
        torch.load(dir_checkpoint,map_location=device)
        )

    # dataset = dataset_superpoint_5k(image_list,feature_list,img_scale, pct_3D_points, crop_size, max_points)
    # infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # n_infer = int(len(dataset))
    val_dataset = dataset_superpoint_5k_online_infer(val_image_list,val_feature_list,img_scale, pct_3D_points, crop_size, max_points)
    infer_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    n_infer = int(len(val_dataset))

    logging.info('Starting infering:\n'        
    '\tBatch size:       %s\n'        
    '\tInfer size:       %s\n'
    '\tCheckpoints:      %s\n' 
    '\tDevice:           %s\n'          
    , batch_size, n_infer, dir_checkpoint, device.type
    )
    


    run_infer(net,infer_loader,device)

        
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import logging

# from vgg import VGGPerception
# from utils.dataset import dataset_superpoint_5k
# from torch.utils.data import DataLoader

# from torchvision.utils import save_image
# from unet import UNet
# from PIL import Image
# import os
# import numpy as np

# def load_annotations(fname):
#     with open(fname,'r') as f:
#         data = [line.strip().split(' ') for line in f]
#     return np.array(data)

# infer_output_dir = './infer_256/'
# dir_checkpoint = './checkpoints/11.pth'
# base_image_dir = '/home/wangr/invsfm/data'
# base_feature_dir = '/home/wangr/superpoint_resize/resize_data_superpoint_1'
# train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
# train_5k_image_rgb=list(train_5k[:,4])

# image_list=[]

# feature_list=[]
# for i in range(len(train_5k_image_rgb)):
#     temp_image_name=train_5k_image_rgb[i]
#     temp_path=os.path.join(base_image_dir,temp_image_name)
#     image_list.append(temp_path)
#     superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
#     feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))

    


# def run_infer(net,infer_loader,device):
#   for batch in infer_loader:
#     try:
#         with torch.no_grad():
#             net.to(device=device)
#             net.eval()
#             input_features = batch['feature']
#             name_i = batch['name']
#             print(name_i)
#             name_i = name_i[0]
#             name_i = name_i.replace('/','^_^')
#             input_features = input_features.to(device=device, dtype=torch.float32)
#             pred = net(input_features).detach()
#             pred = (pred+1.)*127.5
#             pred = pred.to('cpu')
#             ouput_path = infer_output_dir + name_i+ '.png'
#             save_image_tensor(pred,ouput_path)
#             del pred 
#             torch.cuda.empty_cache()
#     except:
#          with torch.no_grad():
#             net.to(device='cpu')
#             net.eval()
#             input_features = batch['feature']
#             name_i = batch['name']
#             print(name_i)
#             name_i = name_i[0]
#             name_i = name_i.replace('/','^_^')
#             input_features = input_features.to(device='cpu', dtype=torch.float32)
#             pred = net(input_features).detach()
#             pred = (pred+1.)*127.5
#             pred = pred.to('cpu')
#             ouput_path = infer_output_dir + name_i+ '.png'
#             save_image_tensor(pred,ouput_path)
#             del pred 
#             torch.cuda.empty_cache()

# def save_image_tensor(input_tensor, filename):
#     assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

#     # to cpu
#     # input_tensor = input_tensor.to(torch.device('cpu'))
#     save_image(input_tensor, filename, normalize=True)


# if __name__ == '__main__':
#     batch_size = 1
#     img_scale = 1
#     crop_size = 0
#     pct_3D_points = 0
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     net = UNet(n_channels=256, n_classes=1)   # input should be 256, resize to 32 so ram enough
#     net.load_state_dict(
#         torch.load(dir_checkpoint)
#         )

#     dataset = dataset_superpoint_5k(image_list,feature_list,img_scale, pct_3D_points, crop_size)
#     infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
#     n_infer = int(len(dataset))
    
#     run_infer(net,infer_loader,device)
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np

from vgg import VGGPerception
from utils.dataset import InferDataset
from torch.utils.data import DataLoader
from PIL import Image

from torchvision.utils import save_image
from unet import UNet
import os

# infer_output_dir = '/cluster/scratch/jiaqiu/infer_output_13_05/'
infer_output_dir = 'D:/infer_test_best/'
infer_origin_dir = 'D:/infer_origin_best/'
# dir_checkpoint = '/cluster/scratch/jiaqiu/checkpoints_12_5_test/14.pth'
dir_checkpoint = 'D:/scale1_2000/14.pth'


def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

base_image_dir = 'D:/npz_torch_data/'
scale = 1
base_dir = 'D:/resize_data_r2d2_1/'
val_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
val_5k_image_rgb=list(val_5k[:,4])
val_image_list=[]
val_feature_list=[]

for i in range(len(val_5k_image_rgb)):
    temp_image_name=val_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    val_image_list.append(temp_path)
    r2d2_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    val_feature_list.append(os.path.join(base_dir,r2d2_feature_name))


def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename, normalize=True)


if __name__ == '__main__':
    batch_size = 1
    # img_scale = 1
    # crop_size = 0
    # pct_3D_points = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=128, n_classes=1)  
    net.load_state_dict(
        torch.load(dir_checkpoint)
        )
    net.to(device=device)

    # dataset_config = {
    #     'augumentation': {
    #         'crop_size': 256,
    #         'dir_img': 'D:/npz_torch_data/',
    #         # 'dir_img': '/cluster/scratch/jiaqiu/npz_torch_data/',
    #     },
    #     'R2D2': {
    #         'gpu': 0,
    #         'model': './models/r2d2_WAF_N16.pt',
    #         'reliability_thr': 0.7,
    #         'repeatability_thr': 0.7,
    #         'scale_f': 2**0.25,
    #         'min_scale': 0,
    #         'max_scale': 1,
    #         'min_size': 128, 
    #         'max_size': 1024,
    #         'max_keypoints': 2000
    #     }
    # }
    try:
        logging.info('Created checkpoint directory')
        os.mkdir(infer_output_dir)
        os.mkdir(infer_origin_dir)
    except OSError:
        pass

    dataset = InferDataset(val_image_list, val_feature_list, max_points=6000)
    infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    n_infer = int(len(dataset))

    logging.info('Starting infering:\n'        
    '\tBatch size:       %s\n'        
    '\tInfer size:       %s\n'
    '\tCheckpoints:      %s\n' 
    '\tDevice:           %s\n'          
    , batch_size, n_infer, dir_checkpoint, device.type
    )
    # upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    for batch in infer_loader:
        # input_features = batch['feature']
        # index = batch['index']
        # input_features = input_features.to(device=device, dtype=torch.float32)
        # pred = net(input_features)
        # cpred = (pred+1.)*127.5
        # ouput_path = infer_output_dir + str(index) + '.png'
        # save_image_tensor(cpred,ouput_path)
        try:
            with torch.no_grad():
                net.to(device=device)
                net.eval()
                input_features = batch['feature']
                image = batch['image']
                index = batch['index']
                input_features = input_features.to(device=device, dtype=torch.float32)
                pred = net(input_features).detach()
                pred = (pred+1.)*127.5
                # pred = upsample(pred)
                ouput_path = infer_output_dir + str(index) + '.png'
                origin_path = infer_origin_dir + str(index) + '.png'
                save_image_tensor(pred,ouput_path)
                save_image_tensor(image,origin_path)
                del pred 
                torch.cuda.empty_cache()
        except:
            with torch.no_grad():
                net.to(device='cpu')
                net.eval()
                input_features = batch['feature']
                image = batch['image']
                index = batch['index']
                input_features = input_features.to(device='cpu', dtype=torch.float32)
                pred = net(input_features).detach()
                pred = (pred+1.)*127.5
                # pred = upsample(pred)
                ouput_path = infer_output_dir + str(index) + '.png'
                origin_path = infer_origin_dir + str(index) + '.png'
                save_image_tensor(pred,ouput_path)
                save_image_tensor(image,origin_path)
                del pred 
                torch.cuda.empty_cache()
        

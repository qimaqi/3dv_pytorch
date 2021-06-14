import unet
from unetpp.Unet_pluspluls import UNet_Nested
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from vgg import VGGPerception
from utils.dataset import InferDataset
from torch.utils.data import DataLoader
from PIL import Image

from torchvision.utils import save_image
# from unet import UNet
# from unet import InvNet
from unetpp import UNet_Nested
import os

# infer_output_dir = '/cluster/scratch/jiaqiu/infer_output_13_05/'
infer_output_dir = 'D:/pp_6000/'
# infer_origin_dir = 'D:/infer_origin/'
dir_checkpoint = 'D:/checkpoints_pp_6000/9.pth'






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

    net = UNet_Nested(n_channels=128, n_classes=1)  
    net.load_state_dict(
        torch.load(dir_checkpoint)
        )
    net.to(device=device)

    dataset_config = {
        'augumentation': {
            'crop_size': 256,
            'dir_img': 'D:/npz_torch_data/',
            # 'dir_img': '/cluster/scratch/jiaqiu/npz_torch_data/',
        },
        'R2D2': {
            'gpu': 0,
            'model': './models/r2d2_WAF_N16.pt',
            'reliability_thr': 0.7,
            'repeatability_thr': 0.7,
            'scale_f': 2**0.25,
            'min_scale': 0,
            'max_scale': 1,
            'min_size': 256, 
            'max_size': 1024,
            'max_keypoints': 6000
        }
    }
    try:
        logging.info('Created checkpoint directory')
        os.mkdir(infer_output_dir)
        # os.mkdir(infer_origin_dir)
    except OSError:
        pass

    dataset = InferDataset(dataset_config)
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
                # origin_path = infer_origin_dir + str(index) + '.png'
                save_image_tensor(pred,ouput_path)
                # save_image_tensor(image,origin_path)
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
                # origin_path = infer_origin_dir + str(index) + '.png'
                save_image_tensor(pred,ouput_path)
                # save_image_tensor(image,origin_path)
                del pred 
                torch.cuda.empty_cache()
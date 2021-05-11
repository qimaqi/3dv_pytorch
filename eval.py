import torch
import torch.nn.functional as F
#from tqdm import tqdm
import torch.nn as nn
from vgg import VGGPerception
from torchvision.utils import save_image
import pytorch_ssim
import numpy as np
import os

def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename, normalize=True)



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    pixel_criterion = nn.L1Loss()       ##### QM: only L1 loss problem: image and feature not match
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    ssim_loss = pytorch_ssim.SSIM()
    l2_loss = nn.MSELoss()
    pix_loss_wt = 1
    per_loss_wt = 5
    sum_pix_loss = 0
    sum_per_loss = 0
    sum_ssim_loss = 0

    output_dir = '/cluster/scratch/jiaqiu/debug_output_eval_online_11_5/'
    img_dir = '/cluster/scratch/jiaqiu/debug_images_eval_online_11_5/'
    try:
        os.mkdir(output_dir)
        os.mkdir(img_dir)
        print('Created eval directory')
    except OSError:
        pass

    global_step = 0 
    #with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        #imgs, true_masks = batch['image'], batch['mask']
        input_features = batch['feature']
        true_imgs = batch['image']
        input_features = input_features.to(device=device, dtype=torch.float32)
        true_imgs = true_imgs.to(device=device, dtype=mask_type)

        with torch.no_grad():
            pred = net(input_features)
            cpred = (pred+1.)*127.5
            P_pred = percepton_criterion(cpred)
            P_img = percepton_criterion(true_imgs)

        perception_loss = ( l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
        pixel_loss = pixel_criterion(cpred/255,true_imgs/255)
        ssim_out = -ssim_loss(cpred, true_imgs)
        ssim_value = - ssim_out.item()
        sum_pix_loss += pixel_loss
        sum_per_loss += perception_loss
        sum_ssim_loss += ssim_value
        tot += pixel_loss*pix_loss_wt + perception_loss*per_loss_wt

        # debug part
        
        tmp_output_dir = output_dir + str(global_step) + '.png'
        tmp_img_dir = img_dir + str(global_step) + '.png'
        save_image_tensor(cpred,tmp_output_dir)
        save_image_tensor(true_imgs,tmp_img_dir)

        global_step += 1

    net.train()
    print('Coarsenet pixel_loss: ',(sum_pix_loss/n_val), 'Coarsenet perception_loss:', sum_per_loss/n_val )
    print('SSIM Value: ',(sum_ssim_loss/n_val))
    return tot / n_val

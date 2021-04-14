import torch
import torch.nn.functional as F
#from tqdm import tqdm
import torch.nn as nn
from vgg import VGGPerception



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    pixel_criterion = nn.L1Loss()       ##### QM: only L1 loss problem: image and feature not match
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()
    pix_loss_wt = 1
    per_loss_wt = 1
    sum_pix_loss = 0
    sum_per_loss = 0


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
        pixel_loss = pixel_criterion(cpred,true_imgs)
        sum_pix_loss += pixel_loss
        sum_per_loss += perception_loss
        tot += pixel_loss*pix_loss_wt + perception_loss*per_loss_wt


    net.train()
    print('Coarsenet pixel_loss: ',(sum_pix_loss/n_val), 'Coarsenet perception_loss:', sum_per_loss/n_val )
    return tot / n_val

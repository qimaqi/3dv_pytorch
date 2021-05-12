import torch
import torch.nn.functional as F
#from tqdm import tqdm
import torch.nn as nn
from vgg import VGGPerception
from torchvision.utils import save_image





def eval_refinenet(refine_net, D,coarse_net,loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    refine_net.eval()
    D.eval()

    n_val = len(loader)  # the number of batch
    tot = 0

    pixel_criterion = nn.L1Loss()       ##### QM: only L1 loss problem: image and feature not match
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()
    pix_loss_wt = 1
    per_loss_wt = 5
    adv_loss_wt = 1
    sum_pix_loss = 0
    sum_per_loss = 0
    sum_adv_loss = 0

    global_step = 0 
    #with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        #imgs, true_masks = batch['image'], batch['mask']
        with torch.no_grad():
            coarse_input_features = batch['feature']
            true_imgs = batch['img_rgb']
            coarse_input_features = coarse_input_features.to(device=device, dtype=torch.float32)
            coarse_pred = coarse_net(coarse_input_features)
            refine_input = torch.cat((coarse_pred,coarse_input_features),axis=1)
            refine_input = refine_input.to(device=device, dtype=torch.float32)
            cr_loss = nn.CrossEntropyLoss()

            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = refine_net(refine_input)  # ##### check the max and min
            rpred = (pred+1.)*127.5
            P_pred = percepton_criterion(rpred)
            P_img = percepton_criterion(true_imgs)
            perception_loss = ( l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
            temp_fake = torch.cat((refine_input,rpred/255,P_pred[0]),axis=1)
            D_fake_input=[temp_fake,P_pred[1],P_pred[2]]
            D_fake = D(D_fake_input)
            n_cha,_,_,_=refine_input.shape
            dgt1 = torch.ones(n_cha, dtype=torch.long)
            dgt1 = dgt1.to(device,dtype=torch.long)
            radvloss = cr_loss(D_fake,dgt1)
            pixel_loss = pixel_criterion(rpred,true_imgs)
            sum_pix_loss += pixel_loss
            sum_per_loss += perception_loss
            sum_adv_loss += radvloss
        tot +=  pixel_loss*pix_loss_wt + perception_loss*per_loss_wt + radvloss*adv_loss_wt


        global_step += 1

    refine_net.train()
    D.train()
    print('refinenet pixel_loss: ',(sum_pix_loss/n_val), 'refinenet perception_loss:', sum_per_loss/n_val , 'refinenet discriminator loss:', sum_adv_loss/n_val )
    return tot / n_val

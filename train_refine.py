# Edited By Qi Ma 
# qimaqi@student.ethz.ch

import argparse
import logging
import os
import sys

from torchvision.utils import save_image
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from eval_refinenet import eval_refinenet
#from unet import InvNet
from unet import UNet
from unet.Discriminator import Discriminator
# from unet import discriminator_loss

from utils.dataset import dataset_r2d2_5k
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from vgg import VGGPerception

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# To do
# delete useless code and make it clear
# to use logging and attribute feature 
# infer to test the result
def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

# dir_checkpoint = './checkpoints/'
# # dir_depth = '../data/nyu_v1_depth/'
# # dir_pos = '../data/nyu_v1_pos/'
# base_image_dir = '/home/wangr/invsfm/data'
# base_feature_dir = '/home/wangr/superpoint_resize/resize_data_superpoint_1'
# train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/train.txt'))

base_image_dir='/cluster/scratch/jiaqiu/npz_torch_data/'
dir_refine_checkpoint = '/cluster/scratch/jiaqiu/checkpoints_06_05_refine/'
dir_d_checkpoint = '/cluster/scratch/jiaqiu/checkpoints_06_05_d/'
train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/train.txt'))
base_feature_dir = '/cluster/scratch/jiaqiu/resize_data_r2d2_0.8/'

train_5k_image_rgb=list(train_5k[:,4])
image_list=[]
feature_list=[]

for i in range(len(train_5k_image_rgb)):
    temp_image_name=train_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    image_list.append(temp_path)
    r2d2_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    feature_list.append(os.path.join(base_feature_dir,r2d2_feature_name))





def train_net(refine_net,
              D,
              coarse_net,
              device,
              pct_3D_points,
              crop_size,
              per_loss_wt,
              pix_loss_wt,
              adv_loss_wt,
              epochs=10,
              batch_size=8,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale = 1):
    
    img_scale = 0.8
    max_points = 2000

    dataset = dataset_r2d2_5k(image_list,feature_list, max_points, crop_size, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batch_size = batch_size
    val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info('Starting training:\n'
                 '\t Epochs:          %s\n'
                 '\tBatch size:       %s\n'
                 '\tLearning rate:    %s\n'
                 '\tTraining size:    %s\n'
                 '\tValidation size:  %s\n'
                 '\tCheckpoints:      %s\n'
                 '\tDevice:           %s\n'
                 '\tImages scaling:   %s\n'
                 '\tCrop Size:        %s\n'
                 , epochs, batch_size, lr, n_train, n_val, save_cp, device.type, img_scale, crop_size
                 )


    optimizer_r = optim.Adam(refine_net.parameters(), lr=lr, eps = 1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_r, milestones=[5,12,16], gamma=0.1)
    optimizer_d = optim.Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999))

    pixel_criterion = nn.L1Loss()
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()

    #if net.n_classes > 1:    # RGB need to reform
    #    criterion = nn.CrossEntropyLoss()
    #else:
    #    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        refine_net.train()
        D.train()

        epoch_loss = 0
        epoch_loss_d = 0
        for batch in train_loader:
            coarse_input_features = batch['feature']
            true_imgs = batch['img_rgb']
            assert coarse_input_features.shape[1] == coarsenet.n_channels, 'Channel match problem'
            coarse_input_features = coarse_input_features.to(device=device, dtype=torch.float32)
            coarse_net.eval()
            coarse_pred = coarse_net(coarse_input_features)
            refine_input = torch.cat((coarse_pred,coarse_input_features),axis=1)
            refine_input = refine_input.to(device=device, dtype=torch.float32)


            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = refine_net(refine_input).detach()  # ##### check the max and min
            rpred = (pred+1.)*127.5


            P_pred = percepton_criterion(rpred)
            P_img = percepton_criterion(true_imgs)
            temp_fake = torch.cat((refine_input,rpred/255,P_pred[0]),axis=1)
            D_fake_input=[temp_fake,P_pred[1],P_pred[2]]

            temp_real = torch.cat((refine_input,true_imgs/255,P_img[0]),axis=1)
            D_real_input=[temp_real,P_img[1],P_img[2]]
            n_cha,_,_,_=refine_input.shape

            D_fake = D(D_fake_input)
            D_real = D(D_real_input)

            dgt0 = torch.zeros(n_cha, dtype=torch.long)
            dgt1 = torch.ones(n_cha, dtype=torch.long)

            dgt  = torch.cat((dgt0,dgt1),axis=0)
            dgt = dgt.to(device, dtype=torch.long)
            d_pred = torch.cat((D_fake,D_real),axis=0)
            cr_loss = nn.CrossEntropyLoss()


            dgt1 = dgt1.to(device,dtype=torch.long)


            for param in D.parameters():
                param.requires_grad = True
            for param in refine_net.parameters():
                param.requires_grad = False

            dloss = cr_loss(d_pred,dgt)
            optimizer_d.zero_grad()
            dloss.backward()
            optimizer_d.step()

            ## freeze dis
            for param in D.parameters():
                param.requires_grad = False
            for param in refine_net.parameters():
                param.requires_grad = True
            coarse_input_features = batch['feature']
            true_imgs = batch['img_rgb']
            coarse_input_features = coarse_input_features.to(device=device, dtype=torch.float32)
            coarse_pred = coarse_net(coarse_input_features)
            refine_input = torch.cat((coarse_pred,coarse_input_features),axis=1)
            refine_input = refine_input.to(device=device, dtype=torch.float32)


            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = refine_net(refine_input)  # ##### check the max and min
            rpred = (pred+1.)*127.5
            P_pred = percepton_criterion(rpred)
            perception_loss = (l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
            temp_fake = torch.cat((refine_input,rpred/255,P_pred[0]),axis=1)
            D_fake_input=[temp_fake,P_pred[1],P_pred[2]]
            D_fake = D(D_fake_input)
            radvloss = cr_loss(D_fake,dgt1)
            pixel_loss = pixel_criterion(rpred/255,true_imgs/255)
            loss = pixel_loss*pix_loss_wt + perception_loss*per_loss_wt + radvloss*adv_loss_wt
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            optimizer_r.zero_grad()
            loss.backward()
            optimizer_r.step()



            global_step += 1


            if global_step % (n_train // (10 * batch_size)) == 0:
                for tag, value in refinenet.named_parameters():
                    tag = tag.replace('.', '/')
                    #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_refinenet(refinenet,D,coarse_net, val_loader, device)
                writer.add_scalar('Loss/test', val_score, epoch)
                scheduler.step()
                print('refine net score: ',(val_score), 'in epoch', epoch )
                #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if save_cp:
            try:
                os.mkdir(dir_refine_checkpoint)
                os.mkdir(dir_d_checkpoint)

                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(refinenet.state_dict(),
                       dir_refine_checkpoint + str(epoch+1) + '.pth')
            torch.save(D.state_dict(),
                       dir_d_checkpoint + str(epoch+1) + '.pth')
            logging.info('Checkpoint %s saved! ',epoch+1)

#writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=24,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a pretrain .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument("--input_attr", metavar='Att' type=str, default='super', choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
    #                help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")
    parser.add_argument("--crop_size", type=int, default=256,     #ƒde to do
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--pct_3D_points", type=lambda s: [float(i) for i in s.split(',')][:2], default=[5.,100.],     # to do
                        help="float,float: Min and max percent of 3D points to keep when performing random subsampling for data augmentation "+ \
                             "(default: 5.,100.)")
    parser.add_argument("--per_loss_wt", type=float, default=5.0, help="%(type)s: Perceptual loss weight (default: %(default)s)")
    parser.add_argument("--pix_loss_wt", type=float, default=1.0, help="%(type)s: Pixel loss weight (default: %(default)s)")
    parser.add_argument("--adv_loss_wt", type=float, default=1.0, help="%(type)s: Discriminator weight (default: %(default)s)")
    parser.add_argument("--max_iter", type=int, default=1e6, help="%(type)s: Stop training after MAX_ITER iterations (default: %(default)s)")
    parser.add_argument("--chkpt_freq", type=int, default=1e4, help="%(type)s: Save model state every CHKPT_FREQ iterations. Previous model state "+ \
                                                                    "is deleted after each new save (default: %(default)s)")
    parser.add_argument("--save_freq", type=int, default=5e4,
                        help="%(type)s: Permanently save model state every SAVE_FREQ iterations "+"(default: %(default)s)")
    parser.add_argument("--val_freq", type=int, default=5e2, help="%(type)s: Run validation loop every VAL_FREQ iterations (default: %(default)s)")
    parser.add_argument("--val_iter", type=int, default=128, help="%(type)s: Number of validation samples per validation loop (default: %(default)s)")


    return parser.parse_args()
########### QM:many parameters need to be used

if __name__ == '__main__':
    load_re = False
    load_dis = False
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s' , device)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    #net = InvNet(n_channels=257, n_classes=1)   # input should be 256, resize to 32 so ram enough
    coarsenet = UNet(n_channels=256, n_classes=1, bilinear=True)

    coarsenet.load_state_dict(torch.load('./coarse.pth', map_location=device))

    coarsenet.to(device=device)

    refinenet = UNet(n_channels=257, n_classes=3, bilinear=True)
    logging.info('Network:\n'
                 '\t %s channels input channels\n'
                 '\t %s output channels (grey brightness)', refinenet.n_channels,  refinenet.n_classes)


    D = Discriminator()



    if load_re:
        refinenet.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        #logging.info(f'Model loaded from {args.load}')
    if load_dis:
        D.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    D.to(device=device)
    refinenet.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(refine_net=refinenet,
                  D=D,
                  coarse_net=coarsenet,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  pct_3D_points = args.pct_3D_points,
                  img_scale=args.scale,
                  crop_size = args.crop_size,
                  per_loss_wt = args.per_loss_wt,
                  pix_loss_wt = args.pix_loss_wt,
                  adv_loss_wt = args.adv_loss_wt,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(refinenet.state_dict(), 'refinenet_INTERRUPTED.pth')
        torch.save(D.state_dict(), 'discriminator_INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
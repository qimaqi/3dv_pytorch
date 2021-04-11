import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import cv2

from eval import eval_net
from unet import InvNet
#from unet import LossNetwork

#from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset2
from utils.dataset import BasicDataset3
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from vgg import VGGPerception


vgg16 = models.vgg16(pretrained=True)
dir_img = '../data/nyu_v1_images/'     ####### QM:change data directory path
dir_features = '../data/nyu_v1_features/'
dir_desc = '../data/nyu_v1_desc/'
dir_checkpoint = 'checkpoints/'
dir_depth = '../data/nyu_v1_depth/'
dir_pos = '../data/nyu_v1_pos/'
    
def train_net(net,
              device,
              pct_3D_points,
              crop_size,
              scale_size, 
              per_loss_wt,
              pix_loss_wt,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=False,  ### QM: no checkpoint
              img_scale=0.5):

    #dataset = BasicDataset2(dir_img, dir_depth, dir_features, img_scale)
    dataset = BasicDataset3(dir_img, dir_depth, dir_pos, dir_desc, img_scale, pct_3D_points, crop_size)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    #writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    #criterion = VGGPerceptualLoss()
    pixel_criterion = nn.L1Loss()       ##### QM: only L1 loss problem: image and feature not match
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()
    #torch.nn.functional.l1_loss
    #if net.n_classes > 1:
    #    criterion = nn.CrossEntropyLoss()
    #else:
    #    criterion = nn.BCEWithLogitsLoss()
    #pix_loss_wt = 0.5
    #per_loss_wt = 0.5

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                input_features = batch['feature']
                true_imgs = batch['image']
                assert input_features.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {input_features.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                input_features = input_features.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_imgs = true_imgs.to(device=device, dtype=mask_type)

                pred = net(input_features)
                cpred = (pred+1.)*127.5
                #print(np.shape(true_imgs))
                #print(np.shape(cpred))
                
                P_pred = percepton_criterion(cpred)
                P_img = percepton_criterion(true_imgs)
                perception_loss = ( l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
                #print(cpred.size())#([1, 1, 168, 224])
               # print(true_imgs.size()) #([1, 1, 168, 224])
                pixel_loss = pixel_criterion(cpred,true_imgs)
                loss = pixel_loss*pix_loss_wt + perception_loss*per_loss_wt

                epoch_loss += loss.item()
                #writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()

                # total loss = L1 pixel loss + L2 perceptual loss
                #total_loss = pix_loss_wt * pix_loss + per_loss_wt * per_loss

                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(input_features.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        #writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        #writer.add_scalar('Dice/test', val_score, global_step)

                    #writer.add_images('images', imgs, global_step)
                    #if net.n_classes == 1:
                        #writer.add_images('masks/true', true_masks, global_step)
                        #writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    #writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a pretrain .pth file')
    #parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                    help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument("--input_attr", metavar='Att' type=str, default='super', choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
    #                help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")            
    parser.add_argument("--crop_size", type=int, default=256,     # to do
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--scale_size", type=lambda s: [int(i) for i in s.split(',')], default=[296,394,512],    # to do
                        help="int,int,int: Sizes to randomly scale images to before cropping them (default: 296,394,512)")
    parser.add_argument("--pct_3D_points", type=lambda s: [float(i) for i in s.split(',')][:2], default=[5.,100.],     # to do
                        help="float,float: Min and max percent of 3D points to keep when performing random subsampling for data augmentation "+\
                        "(default: 5.,100.)")
    parser.add_argument("--per_loss_wt", type=float, default=0.5, help="%(type)s: Perceptual loss weight (default: %(default)s)")   # to do
    parser.add_argument("--pix_loss_wt", type=float, default=0.5, help="%(type)s: Pixel loss weight (default: %(default)s)")        # to do
    parser.add_argument("--max_iter", type=int, default=1e6, help="%(type)s: Stop training after MAX_ITER iterations (default: %(default)s)")
    parser.add_argument("--chkpt_freq", type=int, default=1e4, help="%(type)s: Save model state every CHKPT_FREQ iterations. Previous model state "+\
                        "is deleted after each new save (default: %(default)s)")   
    parser.add_argument("--save_freq", type=int, default=5e4, 
                        help="%(type)s: Permanently save model state every SAVE_FREQ iterations "+"(default: %(default)s)")
    parser.add_argument("--val_freq", type=int, default=5e2, help="%(type)s: Run validation loop every VAL_FREQ iterations (default: %(default)s)")
    parser.add_argument("--val_iter", type=int, default=128, help="%(type)s: Number of validation samples per validation loop (default: %(default)s)")
    parser.add_argument('-s', '--scale', dest='scale', type=float, default= 1.0 ,
                        help='Downscaling factor of the images')
    
    return parser.parse_args()
########### QM:many parameters need to be used

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Create exp dir if does not exist
    #exp_dir = 'wts/{}/coarsenet'.format(prm.input_attr)
    #os.system('mkdir -p {}'.format(exp_dir))


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = InvNet(n_channels=32, n_classes=1)   # input should be 256, resize to 32 so ram enough
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (grey brightness)')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  pct_3D_points = args.pct_3D_points,
                  img_scale=args.scale,
                  crop_size = args.crop_size,
                  scale_size = args.scale_size,
                  per_loss_wt = args.per_loss_wt,
                  pix_loss_wt = args.pix_loss_wt,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

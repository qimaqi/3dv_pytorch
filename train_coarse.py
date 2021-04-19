import argparse
import logging
import os
import sys

from torchvision.utils import save_image
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from eval import eval_net
#from unet import InvNet
from unet import UNet

#from torch.utils.tensorboard import SummaryWriter
#from utils.dataset import BasicDataset2
from utils.dataset import BasicDatasetR2D2
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from vgg import VGGPerception


# To do
# delete useless code and make it clear
# to use logging and attribute feature 
# infer to test the result


#some default dir need images descripton, pos and depth. Attention this time desc and pos is in json !!!!!!!!!!
dir_img = '/cluster/scratch/jiaqiu/nyu_images/'    
#dir_features = '../data/nyu_v1_features/'
dir_desc = '/cluster/scratch/jiaqiu/nyu_r2d2_desc/'
dir_checkpoint = '/cluster/scratch/jiaqiu/checkpoints_19_04/'
load_dir = '/cluster/scratch/jiaqiu/checkpoints_18_04/9.pth'
dir_depth = '/cluster/scratch/jiaqiu/nyu_depth/'
dir_pos = '/cluster/scratch/jiaqiu/nyu_r2d2_pos/'

def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename,normalize=True)


    
def train_net(net,
              device,
              pct_3D_points,
              crop_size, 
              per_loss_wt,
              pix_loss_wt,
              epochs=10,
              batch_size=8,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale = 1):

    #dataset = BasicDataset2(dir_img, dir_depth, dir_features, img_scale)  #without dataaugumentation and load direct feature npz
    dataset = BasicDatasetR2D2(dir_img, dir_depth, dir_pos, dir_desc, img_scale, pct_3D_points, crop_size)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batch_size = 1
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

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps = 1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1) pytorch 1.01
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,12,16], gamma=0.1)

    pixel_criterion = nn.L1Loss()       
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()

    #if net.n_classes > 1:    # RGB need to reform
    #    criterion = nn.CrossEntropyLoss()
    #else:
    #    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        for batch in train_loader:
            input_features = batch['feature']
            true_imgs = batch['image']
            assert input_features.shape[1] == net.n_channels, 'Channel match problem'

            input_features = input_features.to(device=device, dtype=torch.float32)
            #mask_type = torch.float32
            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = net(input_features)  # ##### check the max and min
            cpred = (pred+1.)*127.5     # 
            
            P_pred = percepton_criterion(cpred)
            P_img = percepton_criterion(true_imgs)   ### check perceptional repeat
            perception_loss = ( l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
            #print(cpred.size())#([1, 1, 168, 224])
            # print(true_imgs.size()) #([1, 1, 168, 224])
            pixel_loss = pixel_criterion(cpred,true_imgs)
            loss = pixel_loss*pix_loss_wt + perception_loss*per_loss_wt

            epoch_loss += loss.item()
            #writer.add_scalar('Loss/train', loss.item(), global_step)

            #pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()

            # total loss = L1 pixel loss + L2 perceptual loss
            #total_loss = pix_loss_wt * pix_loss + per_loss_wt * per_loss

            loss.backward()
            #nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()


            global_step += 1
            # debug part
            #if global_step % (n_train // (10 * batch_size)) == 0:
            #    tmp_output_dir = '/cluster/scratch/qimaqi/debug_output/' +str(global_step) + '.png'
            #    tmp_img_dir = '/cluster/scratch/qimaqi/debug_images/'+ str(global_step) + '.png'
            #    save_image_tensor(cpred,tmp_output_dir)
            #    save_image_tensor(true_imgs,tmp_img_dir)
            #    print('cpred maximum', torch.max(cpred))
            #    print('cpred minimum', torch.min(cpred))
            #    print('true_images maximum', torch.max(true_imgs))
            #    print('true_images minimum', torch.min(true_imgs))

            if global_step % (n_train // (10 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                print('Coarsenet score: ',(val_score), 'in epoch', epoch )
                #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + str(epoch+1) + '.pth')
            logging.info('Checkpoint %s saved! ',epoch+1)

    #writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=load_dir,
                        help='Load model from a pretrain .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.8,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument("--input_attr", metavar='Att' type=str, default='super', choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
    #                help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")            
    parser.add_argument("--crop_size", type=int, default=256,     # to do
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--pct_3D_points", type=lambda s: [float(i) for i in s.split(',')][:2], default=[5.,100.],     # to do
                        help="float,float: Min and max percent of 3D points to keep when performing random subsampling for data augmentation "+\
                        "(default: 5.,100.)")
    parser.add_argument("--per_loss_wt", type=float, default=1.0, help="%(type)s: Perceptual loss weight (default: %(default)s)")   
    parser.add_argument("--pix_loss_wt", type=float, default=1.0, help="%(type)s: Pixel loss weight (default: %(default)s)")        
    parser.add_argument("--max_iter", type=int, default=1e6, help="%(type)s: Stop training after MAX_ITER iterations (default: %(default)s)")
    parser.add_argument("--chkpt_freq", type=int, default=1e4, help="%(type)s: Save model state every CHKPT_FREQ iterations. Previous model state "+\
                        "is deleted after each new save (default: %(default)s)")   
    parser.add_argument("--save_freq", type=int, default=5e4, 
                        help="%(type)s: Permanently save model state every SAVE_FREQ iterations "+"(default: %(default)s)")
    parser.add_argument("--val_freq", type=int, default=5e2, help="%(type)s: Run validation loop every VAL_FREQ iterations (default: %(default)s)")
    parser.add_argument("--val_iter", type=int, default=128, help="%(type)s: Number of validation samples per validation loop (default: %(default)s)")

    
    return parser.parse_args()
########### QM:many parameters need to be used

if __name__ == '__main__':
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
    net = UNet(n_channels=129, n_classes=1, bilinear=True)
    logging.info('Network:\n'
            '\t %s channels input channels\n' 
            '\t %s output channels (grey brightness)', net.n_channels,  net.n_classes)


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        #logging.info(f'Model loaded from {args.load}')

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
                  img_scale=0.8,
                  crop_size = args.crop_size,
                  per_loss_wt = args.per_loss_wt,
                  pix_loss_wt = args.pix_loss_wt,
                  val_percent= args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

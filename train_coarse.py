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

from eval import eval_net
from invnet import InvNet
from unet import UNet
from unetplusplus import UNet_Nested
import pytorch_ssim

from utils.dataset import dataset_superpoint_5k, dataset_superpoint_5k_online
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from vgg import VGGPerception
from torch.utils.tensorboard import SummaryWriter
import time

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)


###### change for your own implementation
dir_checkpoint = '/cluster/scratch/qimaqi/checkpoints_30_5_unet_max_6000_lr1e/'  # The path that you want to save checkpoint
base_image_dir= '/cluster/scratch/qimaqi/data_5k/data'  # The path when you data_5k images are saved
base_feature_dir = '/cluster/scratch/qimaqi/data_5k/save_source_dir/resize_data_superpoint_1' # The path when you data_5k feature are saved (offline mode)


# build image list for training and validation 
train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/train.txt'))
val_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
test_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/test.txt'))
train_5k_image_rgb=list(train_5k[:,4])
val_5k_image_rgb=list(val_5k[:,4])
test_5k_image_rgb=list(test_5k[:,4])

image_list=[]
feature_list=[]
for i in range(len(train_5k_image_rgb)):
    temp_image_name=train_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    image_list.append(temp_path)
    superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))

val_image_list=[]
val_feature_list=[]
for i in range(len(val_5k_image_rgb)):
    temp_image_name=val_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    val_image_list.append(temp_path)
    superpoint_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    val_feature_list.append(os.path.join(base_feature_dir,superpoint_feature_name))
    


def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename,normalize=True)


def train_net(net,
              device,
              img_scale,
              max_points,
              crop_size, 
              per_loss_wt,
              pix_loss_wt,
              epochs=10,
              batch_size=8,
              lr=0.001,
              save_cp=True
              ):

    ########### if you want to use already generate feature then use offline below ##############
    dataset = dataset_superpoint_5k(image_list,feature_list,img_scale, crop_size, max_points)
    val_dataset = dataset_superpoint_5k(val_image_list,val_feature_list,img_scale, crop_size, max_points)
    
    ############## if you want to use superpoint online in parallel to process with data ##############
    # dataset = dataset_superpoint_5k_online(image_list,feature_list,img_scale, pct_3D_points, crop_size, max_points)  
    # val_dataset = dataset_superpoint_5k_online(val_image_list,val_feature_list,img_scale, pct_3D_points, crop_size, max_points)
    n_train = len(dataset)
    n_val = len(val_dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batch_size = batch_size
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    writer = SummaryWriter()
    global_step = 0

    logging.info('Starting training:\n'
        '\t Epochs:          %s\n'        
        '\tBatch size:       %s\n'     
        '\tLearning rate:    %s\n' 
        '\tTraining size:    %s\n'  
        '\tValidation size:  %s\n'
        '\tCheckpoints:      %s\n' 
        '\tDevice:           %s\n' 
        '\tCrop Size:        %s\n'
        , epochs, batch_size, lr, n_train, n_val, save_cp, device.type, crop_size
        )

    ########## different optimizer and learning rate scheduler strategy ##############

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps = 1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1) pytorch 1.01
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,12,16], gamma=0.1)

    pixel_criterion = nn.L1Loss()       
    percepton_criterion = VGGPerception()
    percepton_criterion.to(device=device)
    l2_loss = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        for batch in train_loader:
            input_features = batch['feature']
            true_imgs = batch['image']
            assert input_features.shape[1] == net.n_channels, 'Channel match problem'

            input_features = input_features.to(device=device, dtype=torch.float32)
            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = net(input_features)  
            cpred = (pred+1.)*127.5     # 0-255
            
            P_pred = percepton_criterion(cpred)
            P_img = percepton_criterion(true_imgs)  
            perception_loss = ( l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3

            pixel_loss = pixel_criterion(cpred/255,true_imgs/255) 
            ssim_out = -ssim_loss(cpred, true_imgs)
            ssim_value = - ssim_out.item()
            loss = pixel_loss*pix_loss_wt + perception_loss*per_loss_wt + ssim_out

            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)


            optimizer.zero_grad()


            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()


            global_step += 1
            if global_step % (n_train // (5 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                print('Coarsenet score: ',(val_score), 'in epoch', epoch )
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('total_loss', val_score, global_step)
                writer.add_images('output', cpred, global_step)
                writer.add_images('true images', true_imgs, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if True:#(epoch+1)%1==0:
                torch.save(net.state_dict(),
                        dir_checkpoint + str(epoch+1) + '.pth')
                logging.info('Checkpoint %s saved! ',epoch+1)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=16,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default= False,# load your pth for example '/cluster/scratch/qimaqi/checkpoints_17_5_unet_max_1000_lr1e-4/5.pth',
                        help='Load model from a pretrain .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')            
    parser.add_argument("--crop_size", type=int, default=256,     # to do
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--image_rescale", type=float, default=1.0,
                        help="choose scale to resize images in data augumentation")
    parser.add_argument("--max_points", type=int, default=6000,
                        help="maximum feature used for reconstruction")
    parser.add_argument("--per_loss_wt", type=float, default=5.0, help="%(type)s: Perceptual loss weight (default: %(default)s)")   # 5 perceptual loss
    parser.add_argument("--pix_loss_wt", type=float, default=1.0, help="%(type)s: Pixel loss weight (default: %(default)s)")        # 1 perceptual loss do not recommend to change
    parser.add_argument("--feature", type=str, default='Superpoint', help="%(type)s: R2D2 or Superpoint (default: %(default)s)")           
    parser.add_argument("--output", type=int, default=1, help="%(type)s: output 1 is greyscale and output 3 is RGB (default: %(default)s)")           

    
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s' , device)

    # no greyscale or depth here
    if args.feature == 'Superpoint':
        input_channel = 256
    elif args.feature == 'R2D2':
        input_channel = 128
    else:
        logging.info('Feature mode: %s is not Superpoint or R2D2' , args.feature)
        sys.exit(0)

    output_channel = args.output
    assert output_channel == 1 or output_channel == 3, 'output channel is not grey or RGB'

    ########### change here if you want to change different model ##############
    # net = InvNet(n_channels=256, n_classes=1)    
    # net = UNet_Nested(n_channels=input_channel, n_classes=output_channel)
     
    net = UNet(n_channels=input_channel, n_classes=output_channel, bilinear=True)
    logging.info('Network: Unet with SSIM \n'
            '\t %s Max points used\n' 
            '\t %s channels input channels\n' 
            '\t %s output channels (grey brightness)',args.max_points, net.n_channels,  net.n_classes)


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from %s', args.load)

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale= args.image_rescale,
                  max_points = args.max_points,
                  crop_size = args.crop_size,
                  per_loss_wt = args.per_loss_wt,
                  pix_loss_wt = args.pix_loss_wt)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

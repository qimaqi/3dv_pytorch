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
from unet import UNet
# from unetpp import UNet_Nested

from utils.dataset import BasicDatasetR2D2
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from vgg import VGGPerception
from torch.utils.tensorboard import SummaryWriter
import time

# To do
# delete useless code and make it clear
# to use logging and attribute feature 
# infer to test the result

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

#some default dir need images descripton, pos and depth. Attention this time desc and pos is in json !!!!!!!!!!
base_image_dir='/cluster/scratch/jiaqiu/npz_torch_data/'
# base_image_dir = 'D:/npz_torch_data/'
scale = 1
dir_checkpoint = '/cluster/scratch/jiaqiu/checkpoints_3000_small/'
base_dir = '/cluster/scratch/jiaqiu/resize_data_r2d2_' + str(scale) + '/'
# base_dir = 'D:/resize_data_r2d2_1/'
train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/train.txt'))
val_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
train_5k_image_rgb=list(train_5k[:,4])
val_5k_image_rgb=list(val_5k[:,4])
train_image_list=[]
train_feature_list=[]
val_image_list=[]
val_feature_list=[]

for i in range(len(train_5k_image_rgb)):
    temp_image_name=train_5k_image_rgb[i]
    temp_path=os.path.join(base_image_dir,temp_image_name)
    train_image_list.append(temp_path)
    r2d2_feature_name=temp_image_name.replace('/','^_^')+'.npz'
    train_feature_list.append(os.path.join(base_dir,r2d2_feature_name)) 

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
    save_image(input_tensor, filename,normalize=True)


    


def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename,normalize=True)


def train_net(net,
              device,
              max_points,
              feature,
              crop_size, 
              per_loss_wt,
              pix_loss_wt,
              epochs=10,
              batch_size=8,
              lr=0.0001,
              save_cp=True
              ):

    #dataset = BasicDataset2(dir_img, dir_depth, dir_features, img_scale)  #without dataaugumentation and load direct feature npz
    train_dataset = BasicDatasetR2D2(train_image_list, train_feature_list, max_points, crop_size, scale)
    val_dataset = BasicDatasetR2D2(val_image_list, val_feature_list, max_points, crop_size, scale)
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batch_size = 1
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
        '\tFeature used     %s\n'
        , epochs, batch_size, lr, n_train, n_val, save_cp, device.type, crop_size, feature
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
            true_imgs = true_imgs.to(device=device, dtype=torch.float32)

            pred = net(input_features)  # ##### check the max and min
            cpred = (pred+1.)*127.5     # 0-255
            
            P_pred = percepton_criterion(cpred)
            P_img = percepton_criterion(true_imgs)   ### check perceptional repeat
            perception_loss = (l2_loss(P_pred[0],P_img[0]) + l2_loss(P_pred[1],P_img[1]) + l2_loss(P_pred[2],P_img[2])) / 3
            #print(cpred.size())#([1, 1, 168, 224])
            # print(true_imgs.size()) #([1, 1, 168, 224])
            pixel_loss = pixel_criterion(cpred/255,true_imgs/255) 
            
            loss = pixel_loss*pix_loss_wt + perception_loss*per_loss_wt
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            optimizer.zero_grad()


            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.5)
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
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                #print('eval start time',time.ctime())
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                logging.info('Coarsenet score : %s in epoch %s', val_score, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('total_loss', val_score, global_step)
                writer.add_images('output', cpred/255, global_step)
                writer.add_images('true images', true_imgs/255, global_step)

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

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '-c',
#         '--config',
#         type=str,
#         default='configs/train_parameter.yaml',
#         help='config file path')
#     parser.add_argument(
#         '-o',
#         '--override',
#         action='append',
#         default=[],
#         help='config options to be overridden')
#     args = parser.parse_args()
#     return args


def get_args():
    parser = argparse.ArgumentParser(description='Train the CoarseNet on images and correspond superpoint descripton',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=16,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='/cluster/scratch/jiaqiu/checkpoints_3000/9.pth',
                        help='Load model from a pretrain .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')            
    parser.add_argument("--crop_size", type=int, default=256,     # to do
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--pct_points", type=float, default=1.0,
                        help="choose disparse point for reconstruction")
    parser.add_argument("--max_points", type=int, default=3000,
                        help="maximum feature used for reconstruction")
    parser.add_argument("--per_loss_wt", type=float, default=5.0, help="%(type)s: Perceptual loss weight (default: %(default)s)")   
    parser.add_argument("--pix_loss_wt", type=float, default=1.0, help="%(type)s: Pixel loss weight (default: %(default)s)")           
    parser.add_argument("--feature", type=str, default='R2D2', help="%(type)s: R2D2 or Superpoint (default: %(default)s)")           
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

    #net = InvNet(n_channels=257, n_classes=1)   
    # bilinear good or not???
    net = UNet(n_channels=input_channel, n_classes=output_channel)
    logging.info('Network: Unet\n'
            '\t %s channels input channels\n' 
            '\t %s output channels (grey brightness)', net.n_channels,  net.n_classes)


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from %s', args.load)

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  max_points = args.max_points,
                  crop_size = args.crop_size,
                  per_loss_wt = args.per_loss_wt,
                  pix_loss_wt = args.pix_loss_wt,
                  feature = args.feature)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

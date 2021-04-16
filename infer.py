import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from vgg import VGGPerception
from utils.dataset import InferDataset
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from unet import InvNet
from PIL import Image

infer_output_dir = '/cluster/scratch/jiaqiu/infer_output/'
dir_desc = '/cluster/scratch/jiaqiu/nyu_r2d2_desc/'
dir_checkpoint = '/cluster/scratch/jiaqiu/checkpoints/10.pth'
dir_depth = '/cluster/scratch/jiaqiu/nyu_depth/'
dir_pos = '/cluster/scratch/jiaqiu/nyu_r2d2_pos/'
dir_img = '/cluster/scratch/jiaqiu/nyu_images/'


def save_image_tensor(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    # to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename, normalize=True)


if __name__ == '__main__':
    batch_size = 1
    img_scale = 1
    crop_size = 0
    pct_3D_points = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = InvNet(n_channels=129, n_classes=1)   # input should be 256, resize to 32 so ram enough
    net.load_state_dict(
        torch.load(dir_checkpoint)
        )
    net.to(device=device)

    dataset = InferDataset(dir_img, dir_depth, dir_pos, dir_desc, pct_3D_points)
    infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    n_infer = int(len(dataset))

    logging.info('Starting infering:\n'        
    '\tBatch size:       %s\n'        
    '\tInfer size:       %s\n'
    '\tCheckpoints:      %s\n' 
    '\tDevice:           %s\n'          
    , batch_size, n_infer, dir_checkpoint, device.type
    )

    for batch in infer_loader:
        input_features = batch['feature']
        index = batch['index']
        input_features = input_features.to(device=device, dtype=torch.float32)
        pred = net(input_features)
        cpred = (pred+1.)*127.5
        ouput_path = infer_output_dir + str(index) + '.png'
        save_image_tensor(cpred,ouput_path)
        

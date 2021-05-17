# Edited by Qi Ma
# qimaqi@student.ethz.ch

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils import data
from torch.utils.data import Dataset
import logging
from PIL import Image
from utils import data_load
from .superpoint import SuperPoint
from .tools import frame2tensor
from mega_r2d2 import extract_keypoints
import os

# Data basic2 load pos_dir and des_dir to construct a feature 480x640x256 with other point zero
class BasicDataset2(Dataset):
    def __init__(self, dataset_config = {}):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.augumentation_config = dataset_config.get('augumentation')
        self.superpoint = SuperPoint(dataset_config.get('superpoint',{})).eval().to(self.device)
        self.imgs_dir = self.augumentation_config['dir_img']
        self.crop_size = self.augumentation_config['crop_size']
        self.rescale_size = self.augumentation_config['rescale_size']
        

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, rescale_size, crop_size):
        # include rescale, crop and flip, flip set 30% 
        w,h = img.size
        scale_rand_seed_w = torch.rand(1)
        random_scale = rescale_size + (1-rescale_size)*scale_rand_seed_w
        new_w = int(random_scale*w)
        new_h = int(random_scale*h)
        img = img.resize((new_w, new_h), Image.ANTIALIAS)
        assert crop_size <= new_h and crop_size <= new_w,'crop_size is bigger than new rescale image'
        
        img_ = np.array(img)

        crop_rand_seed_w = torch.rand(1)
        crop_rand_seed_h = torch.rand(1)
        crop_w = int(torch.floor((new_w - crop_size) * crop_rand_seed_w))   # 640 - 480 
        crop_h = int(torch.floor((new_h - crop_size) * crop_rand_seed_h))
        img_aug = img_[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]

        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            img_aug = np.flip(img_aug,1)
        
        return img_aug

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg
        keys = ['keypoints', 'scores', 'descriptors']

        #img = data_load.load_img(img_list[i])
        img = Image.open(img_file[0]).convert('L') # read img in greyscale
        img_aug = self.preprocess(img, self.rescale_size, self.crop_size)
        # img_np = np.array(img_aug)
        #frame_tensor = frame2tensor(img_aug, self.device)  # attention here, frame_tensor is ground truth
        frame_tensor = torch.from_numpy(img_aug.copy()).type(torch.FloatTensor)
        last_data = self.superpoint({'image': frame_tensor})
        # last_data = {k: last_data[k] for k in keys} #  ['keypoints', 'scores', 'descriptors']
        keypoints = last_data['keypoints']
        scores = last_data['scores']
        desc = last_data['descriptors']

        keypoints_np = keypoints[0].numpy()
        # scores_np = scores[0].numpy()
        desc_np = desc[0].detach().numpy()
        # print(len(keypoints))
        #print(keypoints_tensor.size())
        #print(scores)

        #keypoints_np = np.array(keypoints)
        #scores_np = np.array(scores)
        #desc_np = np.array(desc)

        # print((keypoints_np[0].size())) #
        #print(np.shape(scores_np)) #
        #print(np.shape(desc_np)) #

        points_num = np.shape(keypoints_np)[1]

        height, width = np.shape(img_aug)  # crop_size x crop_size 
        desc_length = np.shape(desc_np)[0]  # 256 R2D2 is 128

        feature_pad = np.zeros([height,width,desc_length])    # build a 480 x 640 x 256 array   HWC
        for j in range(points_num):
            x = int(keypoints_np[0][j]) #crop_size
            y = int(keypoints_np[1][j]) #crop_size
            feature_pad[y,x,:] = desc_np[:,j]   # to compensate with zero
        
        feature_trans = feature_pad.transpose((2,0,1)) # HWC to CHW
        img_nd = np.expand_dims(img_aug,axis=2)
        img_trans = img_nd.transpose((2,0,1)) # HWC to CHW
        # after preprocess, the feature and image will be well transposed and augumented
        # feature, img = self.preprocess(feature_pad, img, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature_trans.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }

class R2D2_dataset(Dataset):
    def __init__(self, set_type, dataset_config = {}):
        self.set_type = set_type
        self.augumentation_config = dataset_config.get('augumentation')
        self.R2D2 = dataset_config.get('R2D2')
        self.dir_img = self.augumentation_config['dir_img']
        self.crop_size = self.augumentation_config['crop_size']
        self.rescale_size = self.augumentation_config['rescale_size']

        if self.set_type == 'train':
            data=load_annotations(os.path.join(self.dir_img ,'anns/demo_5k/train.txt'))
        else:
            data=load_annotations(os.path.join(self.dir_img ,'anns/demo_5k/val.txt'))
        
        self.image_rgb=list(data[:,4])
        self.ids = list(range(len(self.image_rgb)))
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, img_grey, rescale_size, crop_size):
        # include rescale, crop and flip, flip set 30% 
        w,h = img.size
        scale_rand_seed_w = torch.rand(1)
        random_scale = rescale_size + (1-rescale_size)*scale_rand_seed_w
        new_w = int(random_scale*w)
        new_h = int(random_scale*h)
        img = img.resize((new_w, new_h), Image.ANTIALIAS)
        img_grey = img_grey.resize((new_w, new_h), Image.ANTIALIAS)
        assert crop_size <= new_h and crop_size <= new_w,'crop_size is bigger than new rescale image'
        
        img_ = np.array(img)
        img_grey_ = np.array(img_grey)

        crop_rand_seed_w = torch.rand(1)
        crop_rand_seed_h = torch.rand(1)
        crop_w = int(torch.floor((new_w - crop_size) * crop_rand_seed_w))   # 640 - 480 
        crop_h = int(torch.floor((new_h - crop_size) * crop_rand_seed_h))
        img_aug = img_[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]
        img_grey_aug = img_grey_[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]

        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            img_aug = np.flip(img_aug,1)
            img_grey_aug = np.flip(img_grey_aug,1)
        
        return img_aug, img_grey_aug

    def __getitem__(self, i):
        idx = self.ids[i]
        image_file = os.path.join(self.dir_img, self.image_rgb[idx])     # one image jpg
        #img = data_load.load_img(img_list[i])
        img = Image.open(image_file).convert('RGB') 
        img_grey = Image.open(image_file).convert('L') # read img in greyscale
        img_aug, img_grey = self.preprocess(img, img_grey, self.rescale_size, self.crop_size)
    
        # img_np = np.array(img_aug)
        #frame_tensor = frame2tensor(img_aug, self.device)  # attention here, frame_tensor is ground truth
        # frame_tensor = torch.from_numpy(img_aug.copy()).type(torch.FloatTensor)
        img_aug_cnt = np.ascontiguousarray(img_aug)
        last_data = extract_keypoints(img_aug_cnt, self.R2D2)
        # last_data = {k: last_data[k] for k in keys} #  ['keypoints', 'scores', 'descriptors']
        keypoints_np = last_data['keypoints']
        # scores = last_data['scores']
        desc_np = last_data['descriptors']
        # scores_np = scores[0].numpy()

        # print(len(keypoints))
        #print(keypoints_tensor.size())
        #print(scores)

        #keypoints_np = np.array(keypoints)
        #scores_np = np.array(scores)
        #desc_np = np.array(desc)

        # print((keypoints_np[0].size())) #
        #print(np.shape(scores_np)) #
        #print(np.shape(desc_np)) #

        points_num = np.shape(keypoints_np)[0]

        height, width, _ = np.shape(img_aug)  # crop_size x crop_size 
        desc_length = np.shape(desc_np)[1]  # 256 R2D2 is 128

        feature_pad = np.zeros([height,width,desc_length])    # build a 480 x 640 x 256 array   HWC
        for j in range(points_num):
            x = int(keypoints_np[j][0]) #crop_size
            y = int(keypoints_np[j][1]) #crop_size
            feature_pad[y,x,:] = desc_np[j,:]   # to compensate with zero
        
        feature_trans = feature_pad.transpose((2,0,1)) # HWC to CHW
        img_nd = np.expand_dims(img_grey,axis=2)
        img_trans = img_nd.transpose((2,0,1)) # HWC to CHW
        # after preprocess, the feature and image will be well transposed and augumented
        # feature, img = self.preprocess(feature_pad, img, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature_trans.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }


# use for infer, 
class InferDataset(Dataset):
    def __init__(self, dataset_config = {}):
        self.augumentation_config = dataset_config.get('augumentation')
        self.R2D2 = dataset_config.get('R2D2')
        self.crop_size = self.augumentation_config['crop_size']
        self.dir_img = self.augumentation_config['dir_img']
        data=load_annotations(os.path.join(self.dir_img ,'anns/demo_5k/test.txt'))
        
        self.image_rgb=list(data[:,4])
        self.ids = list(range(len(self.image_rgb)))
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, crop_size):
        w,h = img.size
        min_side = min(w, h)
        if min_side == w:
            img = img.resize((300, int(h * 300 / w)), Image.ANTIALIAS)
        else:
            img = img.resize((int(w * 300 / h), 300), Image.ANTIALIAS)

        new_w, new_h = img.size
        assert crop_size <= new_h and crop_size <= new_w
        img = np.array(img)
        crop_w = int((np.floor(new_w - crop_size)/2))
        crop_h = int((np.floor(new_h - crop_size)/2))
        img_crop = img[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]

        return img_crop

    def __getitem__(self, i):
        idx = self.ids[i]
        image_file = os.path.join(self.dir_img, self.image_rgb[idx])     # one image jpg
        img = Image.open(image_file).convert('RGB') 
        # img = self.preprocess(img, self.scale)
        img = self.preprocess(img, self.crop_size)
        img_cnt = np.ascontiguousarray(img)
        last_data = extract_keypoints(img_cnt, self.R2D2)

        keypoints_np = last_data['keypoints']
        # scores = last_data['scores']
        desc_np = last_data['descriptors']

        points_num = np.shape(keypoints_np)[0]
        height, width, _ = np.shape(img)  # crop_size x crop_size 
        desc_length = np.shape(desc_np)[1]  # 256 R2D2 is 128

        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature_pad = np.zeros([height,width,desc_length])    # build a 480 x 640 x 256 array   HWC
        for j in range(points_num):
            x = int(keypoints_np[j][0]) #crop_size
            y = int(keypoints_np[j][1]) #crop_size
            feature_pad[y,x,:] = desc_np[j,:]   # to compensate with zero

        feature_trans = feature_pad.transpose((2,0,1))
        img_trans = img.transpose((2,0,1))
        
        return {
            'feature': torch.from_numpy(feature_trans.copy()).type(torch.FloatTensor),
            'image' : torch.from_numpy(img_trans.copy()).type(torch.FloatTensor),
            'index': idx  # feature name
        }

class CarvanaDataset(BasicDataset2):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)
# Edited by Qi Ma
# qimaqi@student.ethz.ch

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from utils import data_load
from .superpoint import SuperPoint
from .tools import frame2tensor

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



# Data basic3 load pos_dir and des_dir to construct a feature 480x640x257 with other point zero
class BasicDataset3(Dataset):
    def __init__(self, imgs_dir, depth_dir, pos_dir, desc_dir, scale, pct_3D_points, crop_size):
        self.imgs_dir = imgs_dir
        self.pos_dir = pos_dir
        self.desc_dir = desc_dir
        self.depth_dir = depth_dir
        self.scale = scale
        self.pct_3D_points = pct_3D_points
        self.crop_size = crop_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    #@classmethod
    def preprocess(self, feature, img, scale, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        torch.manual_seed(scale)
        rescale_rand_seed = torch.rand(1)
        random_scale = scale + (1-scale)*rescale_rand_seed 
        new_h = int(h * random_scale)
        new_w = int(w * random_scale)
        if crop_size >=new_h or crop_size >= new_w: crop_size = min(new_h,new_w)
        random_crop = int(crop_size + (min(new_h,new_w)-crop_size)*rescale_rand_seed)
        from torchvision import transforms

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize([new_h,new_w]), #  InterpolationMode.NEAREST, InterpolationMode.BILINEAR and InterpolationMode.BICUBIC
            transforms.RandomCrop(random_crop)
        ])        

        img_nd = np.array(img)
        feature_nd = np.array(feature)
        img_trans = train_transforms(img_nd.copy())
        feature_trans = train_transforms(feature_nd.copy())

        return feature_trans, img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')  # one depth npz
        pos_file = glob(self.pos_dir + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg

        #img = data_load.load_img(img_list[i])
        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        #print(pos_file)

        pos = np.array(data_load.load_json(pos_file[0]))   # 3 x points_num  list, the third is confidence
        desc = np.array(data_load.load_json(desc_file[0]))  # 256 x points_num  list, 256 features

        pos_num = np.shape(pos)[1]
        desc_num = np.shape(desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[0]  # 256 


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length+1 ])    # build a 480 x 640 x 257 array   HWC
        for j in range(pos_num):
            x = int(pos[0][j]) #640
            y = int(pos[1][j]) #480
            feature[y,x,1:] = desc[:,j]   # to compensate with zero
            feature[y,x,0] = (np.array(img)[y,x]/127.5-1)  # only normalize the grey image
        
        self.random_seed = feature.sum()
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img = self.preprocess(feature, img, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': feature,
            'image': img  # ground truth need to be considered
        }


# use for infer, 
class InferDataset(Dataset):
    def __init__(self, imgs_dir, depth_dir, pos_dir, desc_dir, pct_3D_points):
        self.imgs_dir = imgs_dir
        self.pos_dir = pos_dir
        self.desc_dir = desc_dir
        self.depth_dir = depth_dir
        self.pct_3D_points = pct_3D_points

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        #print(h,w,c) # 480, 640, 256
  
        feature_nd = np.array(feature)
        crop_size = 256

        assert crop_size >= 0, 'Crop Size must be positive'
        if crop_size != 0:
            crop_rand_seed_w = torch.rand(1)
            crop_rand_seed_h = torch.rand(1)
            crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))   # 640 - 480 
            crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
            feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        return feature_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')  # one depth npz
        pos_file = glob(self.pos_dir + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg

        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        pos = np.array(data_load.load_json(pos_file[0]))   # 3 x points_num  list, the third is confidence
        desc = np.array(data_load.load_json(desc_file[0]))  # 256 x points_num  list, 256 features

        pos_num = np.shape(pos)[1]
        desc_num = np.shape(desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[0]  # 256 


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length+1 ])    # build a 480 x 640 x 257 array   HWC
        for j in range(pos_num):
            x = int(pos[0][j]) #640
            y = int(pos[1][j]) #480
            feature[y,x,1:] = desc[:,j]   # to compensate with zero
            feature[y,x,1] = (np.array(img)[y,x]/127.5-1)

        feature = self.preprocess(feature)


        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'index': idx  # feature name
        }

class CarvanaDataset(BasicDataset2):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

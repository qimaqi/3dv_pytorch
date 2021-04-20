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

class BasicDataset1(Dataset):
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

    @classmethod
    def preprocess(cls, feature, img, scale, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        #print(h,w,c) # 480, 640, 256
  
        feature_nd = np.array(feature)
        img_nd = np.array(img)

        if len(img_nd.shape) == 2:  # add channel to grey image
            img_nd = np.expand_dims(img_nd, axis=2)  # HWC

        _, _, c2 = np.shape(img_nd) 
        # print(c2) 1
        #print(img_nd.shape)
        #print(feature_nd.shape)

        if scale != 1:
            scale_rand_seed_w = torch.rand(1)
            if scale_rand_seed_w <= 0.5:
                step = (1.0 /scale)  # Attention here, in the outside of images may have lost problem
                
                newW, newH = int(scale * w), int(scale * h)
                #print(step) 
                # print(newW,newH) 512 384
                assert newW > 0 and newH > 0, 'Scale is too small'
                new_img = np.zeros([newH,newW,c2])   
                new_feature = np.zeros([newH,newW,c])
                h_num = 0
                for i in range(h):        
                    w_num = 0
                    if i == int(h_num * step):
                        for j in range(w):
                            if j == int(w_num * step):
                                #print(i,j,'i and j')
                                #print(h_num,w_num,'new h and new w')
                                new_img[h_num,w_num] = img_nd[i,j,:]
                                new_feature[h_num,w_num] = feature_nd[i,j,:]
                                w_num += 1
                        h_num += 1
                w, h = newW, newH
                feature_nd = new_feature
                img_nd = new_img

        # if crop size is 0 then no crop
        assert crop_size >= 0, 'Crop Size must be positive'
        if crop_size != 0:
            crop_rand_seed_w = torch.rand(1)
            crop_rand_seed_h = torch.rand(1)
            crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))   # 640 - 480 
            crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
            feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_nd = img_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            img_nd = np.flip(img_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        img_trans = img_nd.transpose(( 2, 0, 1))    # batch

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
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img = self.preprocess(feature, img, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor)  # ground truth need to be considered
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
        #print(random_crop)
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

class CarvanaDataset(BasicDataset1):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

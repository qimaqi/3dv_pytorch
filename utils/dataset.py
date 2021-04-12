from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


from utils import data_load


# #import cv2 
# def load_json(path):
#     f = open(path,'r')
#     content = f.read()
#     a = json.loads(content)
#     return a.popitem()[1]

###### QM:very important to change
# problem1: the load npz is 640x480x256, np.shape is always 640x480 when image size is 480 x 640
# problem2:
class BasicDataset2(Dataset):
    def __init__(self, imgs_dir, depth_dir, feature_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.feature_dir = feature_dir
        self.depth_dir = depth_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_feature, scale=1):
        #h, w = np.shape(pil_img)[:2] # 480 x 640  ( the size of image is 640x480 when feature of np.array is 640x480, np.array of image is 480x640)
        # h is 480 when w is 640
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))
        feature_nd = np.array(pil_feature)
        feature_nd = np.resize(feature_nd,(224,168,32))  #### QM: resize so ram enough


        #if len(img_nd.shape) == 2:
        #    img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW ... in our feature which is 640x480 WHC -> CHW
        feature_trans = feature_nd.transpose((2, 1, 0)) # channel x 480 x 640
        feature_trans = (feature_trans/127.5)-1
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255

        return feature_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')
        feature_file = glob(self.feature_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        #cv2.imread(path,0)

        assert len(depth_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {depth_file}'
        assert len(feature_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {feature_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        feature = np.load(feature_file[0])['feature']
        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        assert img.size == feature.shape[:2], \
            f'Image and feature {idx} should be the same size, but are {img.size} and {feature.shape[:2]}'

        #img = self.preprocess(img, self.scale)
        feature = self.preprocess(feature, self.scale)   ### QM: the process only transpose channel, need more data augumentation
        img = np.array(img)
        #print(np.shape(img))   # return 480 x 640 height x width, but np.shape return 640x480
        img = np.resize(img,(168,224))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        
        img_trans = img.transpose((2, 0, 1))  # CHWape(img_trans))
        #feature = feature.copy()
        
        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }






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
        logging.info(f'Creating dataset with {len(self.ids)} examples')

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

        #print(img_nd.shape)
        #print(feature_nd.shape)

        if scale != 1:
            step = (1.0 /scale)  # Attention here, in the outside of images may have lost problem
            w_num = 0
            h_num = 0
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            new_img = np.zeros([newH,newW])   
            new_feature = np.zeros([newH,newW,c])
            for i in range(h):
                for j in range(w):
                    if (i == int(h_num * step) and (j == int(w_num * step))):
                        new_img[h_num,w_num] = img_nd[i,j,:]
                        new_feature[h_num,w_num] = feature_nd[i,j,:]
                        w_num += 1
                        h_num += 1
            print('new size',w_num,h_num)
            w, h = newW, newH
            feature_nd = new_feature
            img_nd = new_img

        crop_rand_seed_w = torch.rand(1)
        crop_rand_seed_h = torch.rand(1)
        crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))
        crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
        feature_nd = feature_nd[crop_w:crop_w+crop_size, crop_h:crop_h+crop_size,:]
        img_nd = img_nd[crop_w:crop_w+crop_size, crop_h:crop_h+crop_size,:]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            img_nd = np.flip(img_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        #feature_trans = (feature_trans/127.5)-1   # normalization 127.5 is for RGB, do we need this number here?
        img_trans = img_nd.transpose(( 2, 0, 1))    # batch
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255
        #feature_trans = np.resize(feature_trans,(32 ,168, 224))  #### QM: resize so ram enough
        #img_trans = np.resize(img_trans,(1,168,224))

        return feature_trans, img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')  # one depth npz
        pos_file = glob(self.pos_dir + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg
        #cv2.imread(path,0)
        #print('start get item',idx)

        assert len(depth_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {depth_file}'
        assert len(pos_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {feature_file}'
        assert len(desc_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

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
        feature = np.zeros([height,width,desc_length])    # build a 480 x 640 x 256 array   HWC
        for j in range(pos_num):
            x = int(pos[0][j])
            y = int(pos[1][j])
            feature[y,x] = desc[:,j]   # to compensate with zero
        
        assert np.shape(img) == feature.shape[:2], \
            f'Image and feature {idx} should be the same size, but are {img.size} and {feature.shape[:2]}'

        feature, img = self.preprocess(feature, img, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation
    

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }

class CarvanaDataset(BasicDataset2):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

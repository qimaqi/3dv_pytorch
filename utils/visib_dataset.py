from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


from utils import data_load

class BasicDatasetDepth(Dataset):
    def __init__(self, imgs_dir, gt_depth_dir, proj_depth_dir, pos_dir, desc_dir, scale, pct_3D_points, crop_size):
        self.imgs_dir = imgs_dir
        self.pos_dir = pos_dir
        self.proj_depth_dir = proj_depth_dir
        self.desc_dir = desc_dir
        self.depth_dir = depth_dir
        self.scale = scale
        self.pct_3D_points = pct_3D_points
        self.crop_size = crop_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, depth, scale, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        #print(h,w,c) # 480, 640, 256
  
        feature_nd = np.array(feature)
        depth_nd = np.array(depth)

        if len(depth_nd.shape) == 2: 
            depth_nd = np.expand_dims(depth_nd, axis=2)  # HWC  #needed???

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
                new_img = np.zeros([newH,newW,1])   
                new_feature = np.zeros([newH,newW,c])
                h_num = 0
                for i in range(h):        
                    w_num = 0
                    if i == int(h_num * step):
                        for j in range(w):
                            if j == int(w_num * step):
                                #print(i,j,'i and j')
                                #print(h_num,w_num,'new h and new w')
                                new_depth[h_num,w_num] = depth_nd[i,j,:]
                                new_feature[h_num,w_num] = feature_nd[i,j,:]
                                w_num += 1
                        h_num += 1
                w, h = newW, newH
                feature_nd = new_feature
                depth_nd = new_depth

        crop_rand_seed_w = torch.rand(1)
        crop_rand_seed_h = torch.rand(1)
        crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))   # 640 - 480 
        crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
        feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        depth_nd = depth_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            depth_nd = np.flip(depth_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        #feature_trans = (feature_trans/127.5)-1   # normalization 127.5 is for RGB, do we need this number here?
        depth_trans = depth_nd.transpose(( 2, 0, 1))    # batch
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255
        #feature_trans = np.resize(feature_trans,(32 ,168, 224))  #### QM: resize so ram enough
        #img_trans = np.resize(img_trans,(1,168,224))

        return feature_trans, depth_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')  # one depth npz
        pos_file = glob(self.pos_dir + 'pos_dict_r2d2_' + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + 'desc_dict_r2d2_' + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg
        #cv2.imread(path,0)
        #print('start get item',idx)

        #assert len(depth_file) == 1, \
        #    f'Either no mask or multiple masks found for the ID {idx}: {depth_file}'
        #assert len(pos_file) == 1, \
        #    f'Either no mask or multiple masks found for the ID {idx}: {feature_file}'
        #assert len(desc_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'

        #img = data_load.load_img(img_list[i])
        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        #print(pos_file)

        pos = np.array(data_load.load_json(pos_file[0]))   # points_num x 3  list, the third is scale 
        desc = np.array(data_load.load_json(desc_file[0]))  # points_num x 128   list, 128 features

        pos_num = np.shape(pos)[0]
        desc_num = np.shape(desc)[0]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[1]  # 128


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length+2])    # build a 480 x 640 x 129 array   HWC
        for j in range(pos_num):
            x = int(pos[j][0]) #640
            y = int(pos[j][1]) #480
            feature[y,x,2:] = desc[j,:]   # to compensate with zero
            feature[y,x,1] = (np.array(img)[y,x]/127.5-1)
            feature[y,x,0] = (np.array(depth)[y,x] #add depth
        
        #assert np.shape(img) == feature.shape[:2], \
        #    f'Image and feature {idx} should be the same size, but are {img.size} and {feature.shape[:2]}'

        feature, depth = self.preprocess(feature, depth, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        #print(feature.shape)
        #print(img.shape)

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'depth': torch.from_numpy(depth.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }
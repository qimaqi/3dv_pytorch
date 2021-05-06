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


#BasicDataset2 only for load feature which already 640x480 with 0
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
    def preprocess(cls, feature, img, scale, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        #print(h,w,c) # 480, 640, 256

        # img = img.resize((w, h), Image.ANTIALIAS)

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
        depth_file = glob(self.depth_dir + idx + '.*')
        feature_file = glob(self.feature_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        feature = np.load(feature_file[0])['feature']
        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        feature = self.preprocess(feature, self.scale)   ### QM: the process only transpose channel, need more data augumentation
        img = np.array(img)
        img = np.resize(img,(168,224))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        
        img_trans = img.transpose((2, 0, 1))  # CHWape(img_trans))
        
        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }





# Data basic3 load pos_dir and des_dir to construct a feature 480x640x257 with other point zero
class BasicDataset3(Dataset):
    def __init__(self, imgs_dir, pos_dir, desc_dir, pct_points, max_points, crop_size):
        self.imgs_dir = imgs_dir
        self.pos_dir = pos_dir
        self.desc_dir = desc_dir
        self.pct_points = pct_points
        self.max_points = max_points
        self.crop_size = crop_size
        assert 0 < pct_points <= 1, 'percentage of points must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, img, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC


        if scale != 1:
            w,h = img.size
            new_w = int(scale*w)
            new_h = int(scale*h)
            img = img.resize((new_w, new_h), Image.ANTIALIAS)
            img_rgb = img_rgb.resize((new_w, new_h), Image.ANTIALIAS)

            
        h, w, c = np.shape(feature) 
  
        feature_nd = np.array(feature)
        img_nd = np.array(img)
        img_rgb_nd = np.array(img_rgb)

        if len(img_nd.shape) == 2:  # add channel to grey image
            img_nd = np.expand_dims(img_nd, axis=2)  # HWC

        _, _, c2 = np.shape(img_nd) 

        if crop_size >= h or crop_size >= w:
            crop_size = np.min(h,w)

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
            img_rgb_nd = np.flip(img_rgb_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        img_trans = img_nd.transpose(( 2, 0, 1))    # batch

        return feature_trans, img_trans,img_rgb_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        feature_file=self.feature_list[idx]   
        image_file = self.image_list[idx]  

        #img = data_load.load_img(img_list[i])
   
        img = Image.open(image_file).convert('L')
        img_rgb = Image.open(image_file)

        #print(pos_file)

        temp=np.load(feature_file,allow_pickle=True)
        pos=temp['pts']
        desc=temp['desc']

        pos_num = np.shape(pos)[1]
        desc_num = np.shape(desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[0]  # 256 


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length])    # build a 480 x 640 x 257 array   HWC
        for j in range(pos_num):
            x = int(pos[0][j]) #640
            y = int(pos[1][j]) #480
            feature[y,x,:] = desc[:,j]   # to compensate with zero
  
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img,img_rgb = self.preprocess(feature, img, img_rgb, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor) , # ground truth need to be considered
            'img_rgb': torch.from_numpy(img_rgb.copy()).type(torch.FloatTensor)
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
        pos_file = glob(self.pos_dir + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg

        #img = data_load.load_img(img_list[i])
        img = Image.open(img_file[0]).convert('L') # read img in greyscale
        pos = np.array(data_load.load_json(pos_file[0]))   # 3 x points_num  list, the third is confidence
        desc = np.array(data_load.load_json(desc_file[0]))  # 256 x points_num  list, 256 features

        pos_num = np.shape(pos)[1]
        desc_num = np.shape(desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'

        # choose same max points and use disparse percentage
        if pos_num >= self.max_points:
            new_num = int(self.max_points)
            desc = desc[:,:new_num]
            pos_num = new_num
        
        # pct_points choose
        new_num = int(pos_num * self.pct_points)
        feature_cut = desc[:,:new_num]
        pos_num = new_num

        height, width = np.shape(img)  # (480,640) with scale
        desc_length = np.shape(desc)[0]  # 256 R2D2 is 128

        feature_pad = np.zeros([height,width,desc_length])    # build a 480 x 640 x 256 array   HWC
        for j in range(new_num):
            x = int(pos[0][j]) #640
            y = int(pos[1][j]) #480
            feature_pad[y,x,:] = feature_cut[:,j]   # to compensate with zero
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img = self.preprocess(feature_pad, img, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor)  # ground truth need to be considered
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
        #crop_size = 256

        # assert crop_size >= 0, 'Crop Size must be positive'
        # if crop_size != 0:
        #     crop_rand_seed_w = torch.rand(1)
        #     crop_rand_seed_h = torch.rand(1)
        #     crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))   # 640 - 480 
        #     crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
        #     feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        return feature_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = glob(self.depth_dir + idx + '.*')  # one depth npz
        pos_file = glob(self.pos_dir + 'pos_dict_r2d2_' + idx + '.*')      # one pos json !!! not npz here 
        desc_file = glob(self.desc_dir + 'desc_dict_r2d2_' + idx + '.*')    # one desc json !!! not npz here
        img_file = glob(self.imgs_dir + idx + '.*')     # one image jpg

        depth = np.load(depth_file[0])['depth']
        img = Image.open(img_file[0]).convert('L')

        pos = np.array(data_load.load_json(pos_file[0]))   # 3 x points_num  list, the third is confidence
        desc = np.array(data_load.load_json(desc_file[0]))  # 256 x points_num  list, 256 features

        pos_num = np.shape(pos)[0]
        desc_num = np.shape(desc)[0]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[1]  # 128 


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length+1 ])    # build a 480 x 640 x 129 array   HWC
        for j in range(pos_num):
            x = int(pos[j][0]) #640
            y = int(pos[j][1]) #480
            feature[y,x,1:] = desc[j,:]   # to compensate with zero
            feature[y,x,1] = (np.array(img)[y,x]/127.5-1)

        feature = self.preprocess(feature)


        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'index': idx  # feature name
        }

class CarvanaDataset(BasicDataset2):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

class BasicDatasetR2D2(Dataset):
    def __init__(self, image_list, depth_list, feature_list, pct_points, max_points, crop_size, rescale_list):
        self.image_list = image_list
        self.depth_list = depth_list
        self.feature_list = feature_list
        self.pct_points = pct_points
        self.max_points = max_points
        self.crop_size = crop_size
        self.rescale_list = rescale_list
        assert 0 < pct_points <= 1, 'percentage of points must be between 0 and 1'

        self.ids = list(range(len(image_list)))
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, img, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        feature_nd = np.array(feature)
        img_nd = img.resize((w, h), Image.ANTIALIAS)
        img_nd = np.array(img_nd)

        if len(img_nd.shape) == 2:  # add channel to grey image
            img_nd = np.expand_dims(img_nd, axis=2)  # HWC

        _, _, c2 = np.shape(img_nd) 

        if crop_size >= h or crop_size >= w:
            crop_size = np.min(h,w)

        # if crop size is 0 then no crop
        assert crop_size >= 0, 'Crop Size must be positive'
        if crop_size != 0:
            crop_rand_seed_w = torch.rand(1)
            crop_rand_seed_h = torch.rand(1)
            crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))  
            crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
            feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_nd = img_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            img_nd = np.flip(img_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) 
        img_trans = img_nd.transpose((2, 0, 1))  

        return feature_trans, img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        depth_file = self.depth_list[idx]  
        feature_file=self.feature_list[idx]   
        image_file = self.image_list[idx]

        depth = np.load(depth_file,allow_pickle=True)['image_depth']
        img = Image.open(image_file).convert('L')

        #print(pos_file)
        temp=np.load(feature_file,allow_pickle=True)
        pos=temp['pts']
        desc=temp['desc']

        pos_num = np.shape(pos)[0]
        desc_num = np.shape(desc)[0]
        assert pos_num == desc_num, 'superpoint number matching problem'

        # choose same max points and use disparse percentage
        if pos_num >= self.max_points:
            new_num = int(self.max_points)
            desc = desc[:new_num,:]
            pos_num = new_num
        
        # pct_points choose
        new_num = int(pos_num * self.pct_points)
        feature_cut = desc[:new_num,:]
        pos_num = new_num

        height, width = np.shape(img)
        height = int(height*self.rescale_list[idx])
        width = int(width*self.rescale_list[idx])
        desc_length = np.shape(desc)[1]  # 256 R2D2 is 128
        feature_pad = np.zeros([height,width,desc_length])    
        for j in range(new_num):
            x = int(pos[j][0]) 
            y = int(pos[j][1]) 
            feature_pad[y,x,:] = feature_cut[j,:]   
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img = self.preprocess(feature_pad, img, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor)  # ground truth need to be considered
        }

class dataset_superpoint_5k(Dataset):
    def __init__(self, image_list, feature_list,scale, pct_3D_points, crop_size):
        self.image_list = image_list
        self.feature_list = feature_list
        self.scale = scale
        self.pct_3D_points = pct_3D_points
        self.crop_size = crop_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = list(range(len(image_list)))

        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, img, img_rgb, scale, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC


        if scale != 1:
            w,h = img.size
            new_w = int(scale*w)
            new_h = int(scale*h)
            img = img.resize((new_w, new_h), Image.ANTIALIAS)
            img_rgb = img_rgb.resize((new_w, new_h), Image.ANTIALIAS)

            
        h, w, c = np.shape(feature) 
        #print(h,w,c) # 480, 640, 256
  
        feature_nd = np.array(feature)
        img_nd = np.array(img)
        img_rgb_nd = np.array(img_rgb)

        if len(img_nd.shape) == 2:  # add channel to grey image
            img_nd = np.expand_dims(img_nd, axis=2)  # HWC

        _, _, c2 = np.shape(img_nd) 

        # if crop size is 0 then no crop
        assert crop_size >= 0, 'Crop Size must be positive'
        if crop_size != 0:
            crop_rand_seed_w = torch.rand(1)
            crop_rand_seed_h = torch.rand(1)
            crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))   # 640 - 480 
            crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
            feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_nd = img_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_rgb_nd = img_rgb_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            img_nd = np.flip(img_nd,1)
            img_rgb_nd = np.flip(img_rgb_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        img_trans = img_nd.transpose(( 2, 0, 1))    # batch
        img_rgb_nd = img_rgb_nd.transpose(( 2, 0, 1))

        return feature_trans, img_trans,img_rgb_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        feature_file=self.feature_list[idx]   
        image_file = self.image_list[idx]  

        #img = data_load.load_img(img_list[i])
   
        img = Image.open(image_file).convert('L')
        img_rgb = Image.open(image_file)

        #print(pos_file)

        temp=np.load(feature_file,allow_pickle=True)
        pos=temp['pts']
        desc=temp['desc']

        pos_num = np.shape(pos)[1]
        desc_num = np.shape(desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img)  # 480,640
        desc_length = np.shape(desc)[0]  # 256 


        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature = np.zeros([height,width,desc_length])    # build a 480 x 640 x 257 array   HWC
        for j in range(pos_num):
            x = int(pos[0][j]) #640
            y = int(pos[1][j]) #480
            feature[y,x,:] = desc[:,j]   # to compensate with zero
  
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img,img_rgb = self.preprocess(feature, img, img_rgb, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor) , # ground truth need to be considered
            'img_rgb': torch.from_numpy(img_rgb.copy()).type(torch.FloatTensor)
        }


class dataset_r2d2_5k(Dataset):
    def __init__(self, image_list, feature_list, pct_points, max_points, crop_size, scale_size):
        self.image_list = image_list
        self.feature_list = feature_list
        self.pct_points = pct_points
        self.max_points = max_points
        self.crop_size = crop_size
        self.scale_size = scale_size
        assert 0 < pct_points <= 1, 'percentage of points must be between 0 and 1'

        self.ids = list(range(len(image_list)))
        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, img, img_rgb, crop_size):
        # feature: HWC, img in np shape: HWC. img in size WHC
        h, w, c = np.shape(feature) 
        feature_nd = np.array(feature)
        img_nd = img.resize((w, h), Image.ANTIALIAS)
        img_rgb_nd = img_rgb.resize((w, h), Image.ANTIALIAS)
        img_nd = np.array(img_nd)
        img_rgb_nd = np.array(img_rgb_nd)

        if len(img_nd.shape) == 2:  # add channel to grey image
            img_nd = np.expand_dims(img_nd, axis=2)  # HWC

        _, _, c2 = np.shape(img_nd) 

        if crop_size >= h or crop_size >= w:
            crop_size = np.min(h,w)

        # if crop size is 0 then no crop
        assert crop_size >= 0, 'Crop Size must be positive'
        if crop_size != 0:
            crop_rand_seed_w = torch.rand(1)
            crop_rand_seed_h = torch.rand(1)
            crop_w = int(torch.floor((w - crop_size) * crop_rand_seed_w))  
            crop_h = int(torch.floor((h - crop_size) * crop_rand_seed_h))
            feature_nd = feature_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_nd = img_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
            img_rgb_nd = img_nd[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]

        # random flip
        flip_rand_seed = torch.rand(1)
        if flip_rand_seed <= 0.3:
            feature_nd = np.flip(feature_nd,1)  # left right flip 
            img_nd = np.flip(img_nd,1)
            img_rgb_nd = np.flip(img_rgb_nd,1)

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) 
        img_trans = img_nd.transpose((2, 0, 1))  
        img_rgb_trans = img_nd.transpose((2, 0, 1)) 
        
        return feature_trans, img_trans, img_rgb_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        feature_file=self.feature_list[idx]   
        image_file = self.image_list[idx]

        img = Image.open(image_file).convert('L')
        img_rgb = Image.open(image_file)


        #print(pos_file)
        temp=np.load(feature_file,allow_pickle=True)
        pos=temp['pts']
        desc=temp['desc']

        pos_num = np.shape(pos)[0]
        desc_num = np.shape(desc)[0]
        assert pos_num == desc_num, 'superpoint number matching problem'

        # choose same max points and use disparse percentage
        if pos_num >= self.max_points:
            new_num = int(self.max_points)
            desc = desc[:new_num,:]
            pos_num = new_num
        
        # pct_points choose
        new_num = int(pos_num * self.pct_points)
        feature_cut = desc[:new_num,:]
        pos_num = new_num

        height, width = np.shape(img)
        height = int(height*self.scale_size)
        width = int(width*self.scale_size)
        desc_length = np.shape(desc)[1]  # 256 R2D2 is 128
        feature_pad = np.zeros([height,width,desc_length])    
        for j in range(new_num):
            x = int(pos[j][0]) 
            y = int(pos[j][1]) 
            feature_pad[y,x,:] = feature_cut[j,:]   
        
        # after preprocess, the feature and image will be well transposed and augumented
        feature, img, img_rgb = self.preprocess(feature_pad, img, img_rgb, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img.copy()).type(torch.FloatTensor)  # ground truth need to be considered
            'img_rgb': torch.from_numpy(img_rgb.copy()).type(torch.FloatTensor)
        }
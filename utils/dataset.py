# Edited by Qi Ma
# qimaqi@student.ethz.ch

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.functional import align_tensors
from torch.utils.data import Dataset
import logging
from PIL import Image


from utils import data_load

def border_remove(feature,crop_size):
    for h in range(crop_size):
        for w in range(crop_size):
            if (h<=4) or (w<=4) or (h>=crop_size-4) or ((w>=crop_size-4)):
                feature[h,w,:] = 0
        
    return feature

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

        feature_rnd = border_remove(feature_nd,crop_size)
        # random flip
        # flip_rand_seed = torch.rand(1)
        # if flip_rand_seed <= 0.3:
        #     feature_nd = np.flip(feature_nd,1)  # left right flip 
        #     img_nd = np.flip(img_nd,1)

        # HWC to CHW 
        feature_trans = feature_rnd.transpose((2, 0, 1)) # channel x 480 x 640
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
        h, w, c = np.shape(feature) 
  
        feature_nd = np.array(feature)
        img_nd = np.array(img)

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

        # HWC to CHW 
        feature_trans = feature_nd.transpose((2, 0, 1)) # channel x 480 x 640
        img_trans = img_nd.transpose(( 2, 0, 1))    # batch

        return feature_trans, img_trans

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



class dataset_superpoint_5k(Dataset):
    def __init__(self, image_list, feature_list,scale, pct_3D_points, crop_size, max_points=6000):
        self.image_list = image_list
        self.feature_list = feature_list
        self.scale = scale
        self.pct_3D_points = pct_3D_points
        self.crop_size = crop_size
        self.max_points = max_points
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = list(range(len(image_list)))

        logging.info('Creating dataset with %s examples', len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, feature, img,img_rgb, scale, crop_size):
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
        # flip_rand_seed = torch.rand(1)
        # if flip_rand_seed <= 0.3:
        #     feature_nd = np.flip(feature_nd,1)  # left right flip 
        #     img_nd = np.flip(img_nd,1)
        #     img_rgb_nd = np.flip(img_rgb_nd,1)

        feature_rnd = border_remove(feature_nd,crop_size)

        # HWC to CHW 
        feature_trans = feature_rnd.transpose((2, 0, 1)) # channel x 480 x 640
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

        if pos_num >= self.max_points:
            pos_num = self.max_points

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


class dataset_superpoint_5k_online(Dataset):
    def __init__(self, image_list, feature_list,scale, pct_3D_points, crop_size, max_points=4000):
        self.image_list = image_list
        self.feature_list = feature_list
        self.rescale_size = scale
        self.pct_3D_points = pct_3D_points
        self.crop_size = crop_size
        self.max_points = max_points
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = list(range(len(image_list)))

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
        feature_file=self.feature_list[idx]   
        image_file = self.image_list[idx]  

        #img = data_load.load_img(img_list[i])
   
        img_grey = Image.open(image_file).convert('L')
        img_rgb = Image.open(image_file)
        img_aug, img_grey = self.preprocess(img_rgb, img_grey, self.rescale_size, self.crop_size)
        img_grey_normalized = (img_grey.astype('float32')/255.)
        #print(pos_file)

        # superpoint load and infer
        weights_path = './utils/superpoint_v1.pth'
        nms_dist = 4
        conf_thresh = 0
        nn_thresh = 0.7
        cuda = False
        fe = SuperPointFrontend(weights_path= weights_path,
                          nms_dist= nms_dist,
                          conf_thresh=conf_thresh,
                          nn_thresh=nn_thresh,
                          cuda=cuda)

        pts, desc, _ = fe.run(img_grey_normalized)

        new_pts=pts[:,0:self.max_points]
        new_desc=desc[:,0:self.max_points]

        #temp=np.load(feature_file,allow_pickle=True)
        #pos=temp['pts']
        #desc=temp['desc']

        pos_num = np.shape(new_pts)[1]
        desc_num = np.shape(new_desc)[1]
        assert pos_num == desc_num, 'superpoint number matching problem'
        height, width = np.shape(img_grey)  # 480,640
        desc_length = np.shape(new_desc)[0]  # 256 

        if pos_num >= self.max_points:
            pos_num = self.max_points

        #feature = np.zeros([width,height,desc_length])   # build a 640 x 480 x 256 array
        feature_pad = np.zeros([height,width,desc_length])    # build a 480 x 640 x 257 array   HWC
        for j in range(pos_num):
            x = int(new_pts[0][j]) #640
            y = int(new_pts[1][j]) #480
            feature_pad[y,x,:] = new_desc[:,j]   # to compensate with zero
  
        feature_rb = border_remove(feature_pad,self.crop_size) #remove border
        feature_trans = feature_rb.transpose((2,0,1)) # HWC to CHW
        img_nd = np.expand_dims(img_grey,axis=2)
        img_trans = img_nd.transpose((2,0,1)) # HWC to CHW

        # after preprocess, the feature and image will be well transposed and augumented
        #feature, img,img_rgb = self.preprocess(feature, img, img_rgb, self.scale, self.crop_size)   ### QM: the process only transpose channel, need more data augumentation

        return {
            'feature': torch.from_numpy(feature_trans.copy()).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans.copy()).type(torch.FloatTensor) # ground truth need to be considered
            #'img_rgb': torch.from_numpy(img_rgb.copy()).type(torch.FloatTensor)
        }

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc

class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=True):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
        self.cell = 8 # Size of each output cell. Keep this fixed.
        self.border_remove = 4 # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                     map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int) # Track NMS data.
        inds = np.zeros((H, W)).astype(int) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi) # Softmax.
        dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
        inds = np.argsort(pts[2,:])
        pts = pts[:,inds[::-1]] # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners = True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap
def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)
    
def read_image(impath,resize_scale):
    img = Image.open(impath).convert('L')
    w,h = img.size
    new_w = int(resize_scale*w)
    new_h = int(resize_scale*h)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    grayim = np.array(img)
    grayim = (grayim.astype('float32') / 255.)
    return grayim


class CarvanaDataset(BasicDataset2):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

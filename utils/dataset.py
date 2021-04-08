from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
#import cv2 


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
        #img_nd = np.resize(img_nd,(224,168,32))  #### QM: resize so ram enough


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
        #img = np.resize(img,(170,226))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img_trans = img.transpose((2, 0, 1))  # CHW
        #print(np.shape(img_trans))

        return {
            'feature': torch.from_numpy(feature).type(torch.FloatTensor),
            'image': torch.from_numpy(img_trans).type(torch.FloatTensor)  # ground truth need to be considered
        }






class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

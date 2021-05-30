from scipy.io import loadmat
import json
import argparse
import glob
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch

from scipy.io import savemat

from PIL import Image


def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)
    
def read_image(impath,resize_scale):
    color_img = Image.open(impath)
    w,h = color_img.size
    new_w = int(resize_scale*w)
    new_h = int(resize_scale*h)
    resize_color_img = color_img.resize((new_w, new_h), Image.ANTIALIAS)
    # center_w = int(new_w/2)
    # center_h = int(new_h/2)
    # crop_rectangle = (center_w - 128, center_h -128, center_w+128,center_h+128)  
    # crop_img = resize_color_img.crop(crop_rectangle)
    # crop_gray =  crop_img.convert('L')
    resize_gray_img = resize_color_img.convert('L')

    # return crop_img, crop_gray
    return resize_color_img, resize_gray_img

if __name__ == '__main__':
    ## what you should do
    base_image_dir= '/cluster/scratch/qimaqi/data_5k/' #'/Users/wangrui/Projects/invsfm/'
    save_truth_resize = '/cluster/scratch/qimaqi/data_5k/colorization_val/val_truth/'
    save_gray_resize = '/cluster/scratch/qimaqi/data_5k/colorization_val/val_gray/'
    save_source_dir = '/Users/wangrui/Projects/invsfm/'
    feature_type = 'superpoint'
    resize_scale = 1 ## [0.6, 0.8, 1]
    ##
    # train_5k=load_annotations(os.path.join(base_image_dir,'data/anns/demo_5k/train.txt'))
    # train_5k=train_5k[:,4]
    
    test_5k=load_annotations(os.path.join(base_image_dir,'data/anns/demo_5k/test.txt'))
    test_5k=test_5k[:,4]
    
    val_5k=load_annotations(os.path.join(base_image_dir,'data/anns/demo_5k/val.txt'))
    val_5k=val_5k[:,4]
    image_list=list(val_5k)#(train_5k)  #+list(test_5k)+list(val_5k)
    print('==> Loading pre-trained network.')
    temp_name = 'resize_data_'
    #save_dir = os.path.join(save_source_dir,temp_name+feature_type+'_'+str(resize_scale))

    # if not os.path.exists(save_truth_resize):
    #     os.makedirs(save_truth_resize)

    if not os.path.exists(save_gray_resize):
        os.makedirs(save_gray_resize)

    for i in range(len(image_list)):
        temp=image_list[i].strip()
        test_image=os.path.join(base_image_dir,'data',temp)
        save_name=temp.replace('/','^_^')
        #Get points and descriptors.
        input_image, input_gray = read_image(test_image,resize_scale)

        #image_path = os.path.join(save_truth_resize,save_name+'.jpg')
        #gray_path =os.path.join(save_gray_resize,save_name+'.jpg')
        image_path = os.path.join(save_truth_resize,str(i)+'.jpg')
        gray_path =os.path.join(save_gray_resize,str(i)+'.jpg')

        # input_image.save(image_path)
        input_gray.save(gray_path)


            
     








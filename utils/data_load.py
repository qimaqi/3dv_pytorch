# written by Qi Ma
# qimaqi@student.ethz.cch
# use to load the data descripton and data position as well as image by index

from tqdm import trange
import numpy as np 
import mat73
from PIL import Image
import json
import cv2
import os

img_path = '../nyu_depth_data_labeled.mat'
pos_path = '../desc_dict_all.json'
des_path = '../pos_dict_all.json'
output_path = 'H:/nyu/nyu_v1_images' 

#print(data_dict.keys())

#print(np.shape(images))

def load_mat(path):
    data_dict = mat73.loadmat(path)
    images = data_dict['images']
    return images

def Mat2img():
    for num in trange(np.shape(images)[3]):
    #for num in range(3):
        # Get a new image.
        img = images[:,:,:,num]
        name = str(num) + '.jpg'
        #print(name)
        #cv2.imshow('example',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(output_path,name),img)
        #print('image shape',np.shape(img))
        #img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img2 = img2.astype(np.float32)
        #img3 = img2/255

def load_json(path):
    f = open(path,'r')
    content = f.read()
    a = json.loads(content)
    return a

def load_img(path,mode=0):
    # mode 1 color,0 grey scale, -1 unchange
    img = cv2.imread(path,mode)
    return imag

# def descripton_loader(index):
#     return images[index]

# def descripton_loader(index):
#     return images[index]
    
# #print(np.shape(images))



# img2 = img2.astype(np.float32)
# plt.figure(figsize=(480,640))
# plt.imshow(img2)
# plt.show()
def load_bin_file(fp,dtype,shape):
    data = tf.read_file(fp)
    data = tf.decode_raw(data,dtype)
    data = tf.reshape(data,shape)
    return data
# set scale and crop for data augmentation
def scale_crop(h,w,crxy,crsz,scsz,isval,niter=0):
    scsz = tf.constant(np.float32(scsz),dtype=tf.float32)
    hw = tf.stack([h,w])
    if isval:
        sc = scsz[0]/tf.reduce_min(hw)
        new_sz = tf.to_int32(tf.ceil(sc*hw))
        cry = (new_sz[0]-crsz)//2
        crx = (new_sz[1]-crsz)//2
    else:
        sc = tf.random_shuffle(scsz,seed=niter)[0]/tf.reduce_min(hw)
        new_sz = tf.to_int32(tf.ceil(sc*hw))
        cry = tf.cast(tf.floor(crxy[0]*tf.to_float(new_sz[0]-crsz)),tf.int32)
        crx = tf.cast(tf.floor(crxy[1]*tf.to_float(new_sz[1]-crsz)),tf.int32)
    return sc,new_sz,cry,crx
    
# load batch of image correspondend feature 
def load_proj_bch(pcl_sift_paths,pcl_rgb_paths,
                  crsz,scsz,isval=False,niter=0):
    # input
    # output

    bsz = len(camera_paths)
    proj_depth_batch = []
    proj_sift_batch = []
    proj_rgb_batch = []

    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    crxy = tf.random_uniform([bsz,2],minval=0.,maxval=1.,seed=niter)

    for i in range(bsz):
        # load data from files
        pcl_sift = tf.cast(load_bin_file(pcl_sift_paths[i],tf.uint8,[-1,128]),tf.float32)
        pcl_rgb = tf.cast(load_bin_file(pcl_rgb_paths[i],tf.uint8,[-1,3]),tf.float32)
        sc,_,cry,crx = scale_crop(h,w,crxy[i],crsz,scsz,isval,niter)


        proj_depth = tf.expand_dims(proj_z,axis=1)
        proj_sift = tf.boolean_mask(pcl_sift,mask,axis=0)
        proj_rgb = tf.boolean_mask(pcl_rgb,mask,axis=0)

        # scale pcl
        proj_x = tf.round(proj_x*sc)
        proj_y = tf.round(proj_y*sc)
        h *= sc
        w *= sc

        #################
        # sort proj tensor by depth (descending order)
        _,inds_global_sort = tf.nn.top_k(-1.*proj_z,k=tf.shape(proj_z)[0])
        proj_x = tf.gather(proj_x,inds_global_sort)
        proj_y = tf.gather(proj_y,inds_global_sort)

        # per pixel depth buffer
        seg_ids = tf.cast(proj_x*tf.cast(w,tf.float32) + proj_y, tf.int32)
        data = tf.range(tf.shape(seg_ids)[0])
        inds_pix_sort = tf.unsorted_segment_min(data,seg_ids,tf.reduce_max(seg_ids))
        inds_pix_sort = tf.boolean_mask(inds_pix_sort,tf.less(inds_pix_sort,INT32_MAX))

        proj_depth = tf.gather(tf.gather(proj_depth,inds_global_sort),inds_pix_sort)
        proj_sift = tf.gather(tf.gather(proj_sift,inds_global_sort),inds_pix_sort)
        proj_rgb = tf.gather(tf.gather(proj_rgb,inds_global_sort),inds_pix_sort)

        h = tf.cast(h,tf.int32)
        w = tf.cast(w,tf.int32) 
        proj_yx = tf.cast(tf.concat((proj_y[:,None],proj_x[:,None]),axis=1),tf.int32)
        proj_yx = tf.gather(proj_yx,inds_pix_sort)

        proj_depth = tf.scatter_nd(proj_yx,proj_depth,[h,w,1])
        proj_sift = tf.scatter_nd(proj_yx,proj_sift,[h,w,128])
        proj_rgb = tf.scatter_nd(proj_yx,proj_rgb,[h,w,3])
        ################

        # crop proj
        proj_depth = proj_depth[cry:cry+crsz,crx:crx+crsz,:]
        proj_sift = proj_sift[cry:cry+crsz,crx:crx+crsz,:]
        proj_rgb = proj_rgb[cry:cry+crsz,crx:crx+crsz,:]

        # randomly flip proj
        if not isval:
            proj_depth = tf.image.random_flip_left_right(proj_depth,seed=niter)
            proj_sift = tf.image.random_flip_left_right(proj_sift,seed=niter)
            proj_rgb = tf.image.random_flip_left_right(proj_rgb,seed=niter)

        proj_depth_batch.append(proj_depth)
        proj_rgb_batch.append(proj_rgb)
        proj_sift_batch.append(proj_sift)
    
    return proj_depth_batch, proj_sift_batch, proj_rgb_batch

    # if __name__ == '__main__':
import os, pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

import json
import matplotlib.pyplot as plt



def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)
    
def read_image(impath,resize_scale):
    img = Image.open(impath).convert('RGB')
    w,h = img.size
    new_w = int(resize_scale*w)
    new_h = int(resize_scale*h)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    img = np.array(img)
    return img

def remove_borders(keypoints, descriptors, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask = []
    for i in range(keypoints.shape[0]):
        mask_h = (keypoints[i, 0] >= border) & (keypoints[i, 0] < (height - border))
        mask_w = (keypoints[i, 1] >= border) & (keypoints[i, 1] < (width - border))
        if mask_h & mask_w == 1:
            mask.append(i)
    return keypoints[mask], descriptors[mask], scores[mask]

def extract_keypoints(input_img, config):
    gpu = config['gpu']
    model = config['model']
    reliability_thr = config['reliability_thr']
    repeatability_thr = config['repeatability_thr']
    scale_f = config['scale_f']
    min_scale = config['min_scale'] 
    max_scale = config['max_scale'] 
    min_size = config['min_size']
    max_size = config['max_size'] 
    top_k = config['max_keypoints']

    iscuda = common.torch_set_gpu(gpu)

    # load the network...
    net = load_network(model)
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = reliability_thr, 
        rep_thr = repeatability_thr)
    
    img = input_img
    H, W, _ = img.shape

    img = norm_RGB(img)[None]

    if iscuda: img = img.cuda()

    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, scale_f, min_scale, 
        max_scale, min_size, max_size, verbose = True)
    
    xys = xys.cpu().numpy()
    print(np.shape(xys))
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    border = 4
    xys, desc, scores = remove_borders(xys, desc, scores, border, H, W)
    idxs = scores.argsort()[-top_k or None:]

    print(f"Saving {len(idxs)}")
    keypoints = xys[idxs]
    descriptors = desc[idxs]

    return {
            'keypoints': keypoints,
            'descriptors': descriptors,
        }

# if __name__ == '__main__':

#     model = './models/r2d2_WAF_N16.pt'
#     scale_f = 2**0.25
#     min_size = 256
#     max_size = 1024
#     min_scale = 0
#     max_scale = 1
#     reliability_thr = 0.7
#     repeatability_thr = 0.7
#     gpu = 0
#     top_k = 6000

#     base_image_dir= '/cluster/scratch/jiaqiu/npz_torch_data/'
#     save_source_dir = '/cluster/scratch/jiaqiu/'
#     feature_type = 'r2d2'
#     resize_scale = 0.6 ## [0.6, 0.8, 1]

#     train_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/train.txt'))
#     train_5k=train_5k[:,4]
    
#     test_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/test.txt'))
#     test_5k=test_5k[:,4]
    
#     val_5k=load_annotations(os.path.join(base_image_dir,'anns/demo_5k/val.txt'))
#     val_5k=val_5k[:,4]

#     image_list=list(train_5k)+list(test_5k)+list(val_5k)

#     temp_name = 'resize_data_'
#     save_dir = os.path.join(save_source_dir,temp_name+feature_type+'_'+str(resize_scale))

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     print('start saving data')
#     for i in range(len(image_list)):
#         temp=image_list[i].strip()
#         test_image=os.path.join(base_image_dir, temp)
#         save_name=temp.replace('/','^_^')
#         #Get points and descriptors.
#         input_img = read_image(test_image,resize_scale)

#         keypoints, descriptors = extract_keypoints(input_img, gpu, model, reliability_thr, repeatability_thr, scale_f, min_scale, max_scale, min_size, max_size, top_k)

#         np.savez_compressed(os.path.join(save_dir,save_name), pts=keypoints, desc=descriptors)

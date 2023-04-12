# Gaussian kernel for downsampling
import scipy.ndimage.filters as fi
import numpy as np
import torch.nn.functional as F
import torch 
from torchvision.transforms.functional import to_tensor
import os
from os import listdir
from os.path import join
import torch.utils.data as data
from PIL import Image, ImageOps
import random
from bicubic import imresize
import cv2

def gaussian_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code
    Args:
        x (Tensor, [C, T, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)  # 13 and 0.8 for x2
    elif scale == 3:
        h = gkern(13, 1.2)  # 13 and 1.2 for x3
    elif scale == 4:
        h = gkern(13, 1.6)  # 13 and 1.6 for x4
    else:
        print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(R))
        exit(1)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    # please keep the operation same as training.
    # if  downsample to 32 on training time, use the below code.
    x = x[:, :, 2:-2, 2:-2]
    # if downsample to 28 on training time, use the below code.
    # x = x[:,:,scale:-scale,scale:-scale]
    x = x.view(C, T, x.size(2), x.size(3))
    return x


def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def image_reader():
    scale = 4
    # with open('/home/aagraharibaniya/360VSR/360Videos/Frames/test.txt') as f:
    # scene_list = [line.rstrip() for line in f]
    scene_list = ['calendar','city','foliage','walk']
    for  scene_name in scene_list:
        alist = os.listdir(os.path.join('/home/aagraharibaniya/VSR_Models/BasicSR/datasets/Vid4/GT/', scene_name))
        alist.sort()
        image_filenames = [os.path.join('/home/aagraharibaniya/VSR_Models/BasicSR/datasets/Vid4/GT/', scene_name, x) for x in alist] 
        image_filenames = sorted(image_filenames)
        target = []
        for i in range(len(image_filenames)):
            GT_temp = modcrop(Image.open(image_filenames[i]).convert('RGB'), scale)
            target.append(GT_temp)
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target,dtype=np.uint8)
        # if scale == 4:
        #     target = np.lib.pad(target, pad_width=((0,0), (2*scale,2*scale), (2*scale,2*scale), (0,0)), mode='reflect')
        t, h, w, c = target.shape
        # target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        # target = to_tensor(target) # Tensor, [CT',H',W']
        # target = target.view(c,t,h,w)
        # LR = gaussian_downsample(target)
        # LR = torch.permute(LR,(1,2,3,0)) # [T,H,W,C]
        # # print(LR.shape)
        # LR = LR.numpy()[:,:,:,::-1] 
        # print(LR.shape)
        # target = target.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
        # target = target.cpu().numpy()[:,:,:,::-1]
        if not os.path.exists('/home/aagraharibaniya/VSR_Models/BasicSR/datasets/Vid4/BD/'+str(scene_name)):
                os.makedirs('/home/aagraharibaniya/VSR_Models/BasicSR/datasets/Vid4/BD/'+str(scene_name))
        for i in range(t):
            img_name = image_filenames[i].replace('GT','BD')
            img = target[i]
            # img = cv2.resize(img,(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
            # tr_name = image_filenames[i]
            # g_filter = cv2.getGaussianKernel(13,1.6)
            # img = cv2.pyrDown(target[i],img)
            img = cv2.GaussianBlur(img,(13,13),1.6,cv2.BORDER_DEFAULT)
            img = cv2.resize(img,(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_LINEAR)
            # img = LR[i]*255
            # img = img.astype(np.uint8)
            # img = Image.fromarray(img,mode="RGB")
            # img.save(img_name)
            cv2.imwrite(img_name, img[:,:,::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # cv2.imwrite(img_name, LR[i]*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # cv2.imwrite(img_name, img)


if __name__=='__main__':
    image_reader()
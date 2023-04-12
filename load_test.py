import os
from os import listdir
from os.path import join
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample
import cv2



def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img


class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = os.listdir(os.path.join(image_dir, scene_name))
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, scene_name, x) for x in alist] 
        self.image_filenames = sorted(self.image_filenames)
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
    def __getitem__(self, index):
        target = []
        max = 20
        if self.L > max:
            nFrames = max
        else:
            nFrames = self.L 
        for i in range(nFrames):
            GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            target.append(GT_temp)
        LR = [frame.resize((int(target[0].size[0]/self.scale),int(target[0].size[1]/self.scale)), Image.BICUBIC) for frame in target]
        target = [np.asarray(HR) for HR in target] 
        IN = [np.asarray(IN) for IN in LR]
        target = np.asarray(target)
        IN = np.asarray(IN)
        # if self.scale == 4:
        #     target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t, h, w, c = target.shape
        t_lr, h_lr, w_lr, c_lr = IN.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        IN = IN.transpose(1,2,3,0).reshape(h_lr, w_lr, -1)
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
            IN = self.transform(IN)
        target = target.view(c,t,h,w)
        IN = IN.view(c_lr,t_lr,h_lr,w_lr)
        # LR = gaussian_downsample(target, self.scale) # [c,t,h,w]
        # LR = torch.cat((LR[:,1:2,:,:], LR,LR[:,t-1:t,:,:]), dim=1)
        LR = torch.cat((IN[:,1:2,:,:], IN,IN[:,t_lr-1:t_lr,:,:]), dim=1)
        del IN
        return LR, target
        
    def __len__(self):
        return 1 


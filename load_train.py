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



class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.L = len(alist)
        self.scale = scale
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        HR = []
        max = 20
        if self.L > max:
            nFrames = max
        else:
            nFrames = self.L 
        for i in range(nFrames):
            GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            HR.append(GT_temp)

        if self.data_augmentation:
            HR = [ImageOps.flip(j) for j in HR]

            if random.random() < 0.5:
                HR = [ImageOps.mirror(j) for j in HR]
            
            if random.random() < 0.5:
                HR = [j.rotate(180) for j in HR]
        
        GT = [np.asarray(GT) for GT in HR]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C]
        T,H,W,C = GT.shape
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')
        t, h, w, c = GT.shape
        GT = GT.transpose(1,2,3,0).reshape(h, w, -1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w)
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR,LR[:,t-1:t,:,:]), dim=1)
        
        return LR, GT  
        
        

    def __len__(self):
        return len(self.image_filenames)


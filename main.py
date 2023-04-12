# from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set 
import pdb
from torch.optim import lr_scheduler 
import socket
import time
import cv2
import math
import sys
import numpy as np
from arch import ResConvo
import datetime
import torchvision.utils as vutils
import random

parser = argparse.ArgumentParser(description='PyTorch conv_net')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=145, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots. This is a savepoint, using to save training model.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--file_list', type=str, default='train.txt', help='where record all of image name in dataset.')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--layer', type=int, default=15, help='network layer')
parser.add_argument('--stepsize', type=int, default=10, help='Learning rate is decayed by a factor of 10 every half of total epochs')
parser.add_argument('--gamma', type=float, default=0.1 , help='learning rate decay')
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='Location to save checkpoint models')
parser.add_argument('--weight-decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--log_name', type=str, default='S3PO')
parser.add_argument('--gpu-devices', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 

opt = parser.parse_args()
opt.data_dir = ''
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def main():
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
       use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    pin_memory = True if use_gpu else False

    print(opt)
    print('===> Loading Datasets')
    train_set = get_training_set(opt.data_dir, opt.scale, opt.data_augmentation, opt.file_list) 
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True)
    print('===> DataLoading Finished')
    # Selecting network layer
    n_c = 128
    n_b = 10
    conv_net = ResConvo(opt.scale, n_c, n_b) # initial filter generate network 
    criterion = nn.SmoothL1Loss(reduction='none')
    p = sum(p.numel() for p in conv_net.parameters())/1048576.0
    print('Model Size: {:.2f}M'.format(p))
    print(conv_net)
    print('===> {}L model has been initialized'.format(n_b))
    conv_net = torch.nn.DataParallel(conv_net)
    if use_gpu:
        # conv_net.load_state_dict(torch.load('./result/weight/2022-05-19-11-27/STIFS_3_localX4_15L_64_epoch_100.pth', map_location=lambda storage, loc: storage))
        conv_net = conv_net.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(conv_net.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay) 
    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = opt.stepsize, gamma=opt.gamma)

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        train(train_loader, conv_net, opt.scale, optimizer, epoch, use_gpu, n_c,criterion) #fed data into network
        scheduler.step()
        if (epoch) % (opt.snapshots) == 0:
            checkpoint(conv_net, epoch)

def train(train_loader, conv_net, scale, optimizer, epoch, use_gpu, n_c,criterion):
    train_mode = True
    epoch_loss = 0
    conv_net.train()
    total_loss = 0
    
    for iteration, data in enumerate(train_loader):
        x_input, targets = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        if use_gpu:
            x_input = Variable(x_input).cuda()
            targets = Variable(targets).cuda()
        t0 = time.time()
        optimizer.zero_grad()
        B, _, T, _ ,_ = x_input.shape
        out = []
        init = True
        for i in range(T-2):
            f1 = x_input[:,:,i,:,:]
            f2 = x_input[:,:,i+1,:,:]
            f3 = x_input[:,:,i+2,:,:]
            if init:
                init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])
                init_out = init_temp.repeat(1, scale*scale*3,1,1)
                hid = init_temp.repeat(1, n_c, 1,1)
                hid, prediction = conv_net(f1,f2,f3,hid,init_out,init)
                out.append(prediction)
                init = False
            else:
                hid, prediction = conv_net(f1,f2,f3,hid,prediction,init)
                out.append(prediction)
        predictions = torch.stack(out,dim=2)
        b,_,n,_,_=targets.size()
        loss = criterion(predictions, targets)
        loss = torch.sum(wSmoothL1(predictions,loss))
        loss = loss/(b*n)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader),loss.item(), (t1 - t0)),flush=True)
    print("Epoch-{} Avg Loss {}".format(epoch,(epoch_loss/iteration)))


def checkpoint(conv_net, epoch): 
    save_model_path = os.path.join(opt.save_model_path, systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = opt.log_name+"X"+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(conv_net.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def genERP(i,j,N):
    val = math.pi/N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos( (j - (N/2) + 0.5) * val )
    return w

def compute_map_ws(pred):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    # pred = pred.squeeze(0).permute(1,2,3,0)
    # pred = pred.cpu().detach().numpy()[:,:,:,::-1]
    # map = torch.tensor([1, 1, 376, 496],dtype=torch.float64)
    # print(map.shape,pred.shape)
    a,b,c,d,e = pred.shape
    map = np.zeros((d,e))
    # transform = torchvision.transforms.Lambda(lambda pred: torch.cos((torch.index_select(pred,3,Variable(torch.tensor(range(0,d))).cuda())-(d/2)+0.5)*(math.pi/d)))

    for j in range(0,d):
        for i in range(0,e):
            map[j,i] = genERP(i,j,d)

    return torch.from_numpy(map)

def wSmoothL1(pred,reg_loss):

    map = Variable(compute_map_ws(pred)).cuda()
    # print(map.shape, reg_loss.shape)
    val = torch.mul(reg_loss,map)
    # print('w/o wieight =',np.sum((mx-my)**2))
    # print('weight = ',np.sum(mw))
    # print('before ws-mse=',val)
    # den = val / np.sum(mw)

    return val



if __name__ == '__main__':
    main()    

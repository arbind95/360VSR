from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools
import os
import cv2
import torchvision.utils as vutils

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv1_l = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_l = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2,self.conv1_l, self.conv2_l], 0.1)
    
    def forward(self, x_res, x_loc):
        identity_g = x_res
        out_g = F.relu(self.conv1(x_res), inplace=True)
        out_g = self.conv2(out_g)

        identity_l = x_loc
        out_l = F.relu(self.conv1_l(x_loc), inplace=True)
        out_l = self.conv2_l(out_l)

        out = out_g + out_l

        return out+identity_g,out+identity_l

class Feat_Extract(nn.Module):
    
    def __init__(self, nf=64):
        super(Feat_Extract, self).__init__()
        self.conv_iden1 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        self.conv_iden2 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        self.conv_iden3 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        
        self.conv1_f1 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        self.conv2_f1 = nn.Conv2d(nf, nf, (3,3), stride=(1,1), padding=(1,1))

        self.conv1_f2 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        self.conv2_f2 = nn.Conv2d(nf, nf, (3,3), stride=(1,1), padding=(1,1))

        self.conv1_f3 = nn.Conv2d(3, nf, (3,3), stride=(1,1), padding=(1,1))
        self.conv2_f3 = nn.Conv2d(nf, nf, (3,3), stride=(1,1), padding=(1,1))

        self.feat_set1 = nn.Conv2d(nf*2, 64, (3,3), stride=(1,1), padding=(1,1))
        self.feat_set2 = nn.Conv2d(nf*2, 64, (3,3), stride=(1,1), padding=(1,1))

        self.conv_sa = nn.Conv2d(nf*2,nf,kernel_size=7,bias=False,padding=(3,3))
        self.lrelu = nn.LeakyReLU(0.1,inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_ca_avg1 = nn.Conv2d(nf,32,kernel_size=3,bias=False,padding=(1,1))
        self.conv_ca_avg2 = nn.Conv2d(32,nf,kernel_size=3,bias=False,padding=(1,1))
        
        self.conv_ca_max1 = nn.Conv2d(nf,32,kernel_size=3,bias=False,padding=(1,1))
        self.conv_ca_max2 = nn.Conv2d(32,nf,kernel_size=3,bias=False,padding=(1,1))

        # initialization
        initialize_weights([self.conv_iden1,self.conv_iden2,self.conv_iden3,self.conv1_f1, self.conv2_f1,self.conv1_f2,self.conv2_f2,self.conv1_f3,self.conv2_f3], 0.1)

    def forward(self, f1,f2,f3):
        identity_f1 = self.conv_iden1(f1)
        identity_f2 = self.conv_iden2(f2)
        identity_f3 = self.conv_iden3(f3)

        out_f1 = F.relu(self.conv1_f1(f1), inplace=True)
        out_f1 = self.conv2_f1(out_f1)

        out_f2 = F.relu(self.conv1_f2(f2), inplace=True)
        out_f2 = self.conv2_f2(out_f2)

        out_f3 = F.relu(self.conv1_f3(f3), inplace=True)
        out_f3 = self.conv2_f3(out_f3)

        out = out_f1 + out_f2 + out_f3

        final_feat_set1 = torch.cat((out+identity_f1,out+identity_f2),dim=1)
        final_feat_set1 = F.relu(self.feat_set1(final_feat_set1))

        final_feat_set2 = torch.cat((out+identity_f2,out+identity_f3),dim=1)
        final_feat_set2 = F.relu(self.feat_set2(final_feat_set2))

        final_feat = torch.cat((final_feat_set1,final_feat_set2),dim=1)

        pooled_feat_avg = torch.nn.functional.avg_pool2d(final_feat,kernel_size=3,stride=1,padding=(1,1))
        pooled_feat_max = torch.nn.functional.max_pool2d(final_feat,kernel_size=3,stride=1,padding=(1,1))
        
        pooled_feat = torch.cat((pooled_feat_avg,pooled_feat_max),dim=1)
        feat_sa = self.conv_sa(pooled_feat)
        feat_sa = self.sigmoid(feat_sa)
        final_feat = torch.mul(feat_sa,final_feat)

        pooled_sa_avg = torch.nn.functional.avg_pool2d(final_feat,kernel_size=3,stride=1,padding=(1,1))
        pooled_sa_max = torch.nn.functional.max_pool2d(final_feat,kernel_size=3,stride=1,padding=(1,1))

        feat_ca_avg = self.conv_ca_avg1(pooled_sa_avg)
        feat_ca_avg = self.lrelu(feat_ca_avg)
        feat_ca_avg = self.conv_ca_avg2(feat_ca_avg)
        
        feat_ca_max = self.conv_ca_max1(pooled_sa_max)
        feat_ca_max = self.lrelu(feat_ca_max)
        feat_ca_max = self.conv_ca_max2(feat_ca_max)

        feat_ca = self.sigmoid(feat_ca_max+feat_ca_avg)
        final_feat = torch.mul(feat_ca,final_feat)
        

        return final_feat



class neuro(nn.Module):
    def __init__(self, n_c, n_b, scale):
        super(neuro,self).__init__()
        pad = (1,1)
        block = []
        
        self.feats = Feat_Extract(nf=n_c)
        self.conv_cat = nn.Conv2d(48+n_c+3,n_c,(3,3), stride=(1,1), padding=pad)
        self.recon_trunk1 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk2 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk3 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk4 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk5 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk6 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk7 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk8 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk9 = ResidualBlock_noBN(nf=n_c)
        self.recon_trunk10 = ResidualBlock_noBN(nf=n_c)
        self.conv_res1 = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        self.conv_res2 = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        self.conv_loc1 = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        self.conv_loc2 = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        initialize_weights([self.conv_cat,self.conv_res1,self.conv_res2,self.conv_loc1,self.conv_loc2], 0.1)

    def forward(self, f1,f2,f3,hid,x_out):
        local_feat = F.relu(self.feats(f1,f2,f3))        

        fused_res = torch.cat((f2,hid,x_out),dim=1)
        res_feat = F.relu(self.conv_cat(fused_res))
        
        out_res,out_loc = self.recon_trunk1(res_feat, local_feat)
        out_res,out_loc = self.recon_trunk2(out_res, out_loc)
        out_res,out_loc = self.recon_trunk3(out_res, out_loc)
        out_res,out_loc = self.recon_trunk4(out_res, out_loc)
        out_res,out_loc = self.recon_trunk5(out_res, out_loc)
        out_res,out_loc = self.recon_trunk6(out_res, out_loc)
        out_res,out_loc = self.recon_trunk7(out_res, out_loc)
        out_res,out_loc = self.recon_trunk8(out_res, out_loc)
        out_res,out_loc = self.recon_trunk9(out_res, out_loc)
        out_res,out_loc = self.recon_trunk10(out_res, out_loc)

        hid_out_res = F.relu(self.conv_res1(out_res))
        out_res = self.conv_res2(out_res)
        hid_out_loc = F.relu(self.conv_loc1(out_loc))
        out_loc = self.conv_loc2(out_loc)
        hid = hid_out_res+hid_out_loc
        return out_res,out_loc,hid

class ResConvo(nn.Module):
    def __init__(self, scale, n_c, n_b):
        super(ResConvo, self).__init__()
        self.neuro = neuro(n_c, n_b, scale)
        self.upconv1 = nn.Conv2d(n_c,scale**2*3,(3,3), stride=(1,1), padding=1)
        self.upconv2 = nn.Conv2d(n_c,scale**2*3,(3,3), stride=(1,1), padding=1)
        self.conv_last1 = nn.Conv2d(3*2,3*2,(3,3), stride=(1,1), padding=1)
        self.conv_last2 = nn.Conv2d(3*2,3,(3,3), stride=(1,1), padding=1)
        self.down = PixelUnShuffle(scale)
        self.scale = scale
        initialize_weights([self.upconv1,self.upconv2,self.conv_last1,self.conv_last2], 0.1)
        
    def forward(self, f1,f2,f3,hid, x_out,init):
        if init:
            x_res, x_loc,hid = self.neuro(f1,f2,f3,hid, x_out)
        else:
            x_out = self.down(x_out)
            x_res, x_loc,hid = self.neuro(f1,f2,f3,hid, x_out)
        
        out_res = self.upconv1(x_res)
        out_res = F.pixel_shuffle(out_res, 4)

        out_loc = self.upconv2(x_loc)
        out_loc = F.pixel_shuffle(out_loc, 4)

        out = torch.cat((out_res,out_loc),dim=1)
        out = self.conv_last1(out)
        out = F.relu(out)
        out = self.conv_last2(out)

        out =  out + F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return hid,out



def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
import torch.nn as nn
import torch
import utils
from torchvision.models import vgg16
from PIL import Image
from torch.nn import functional as F
import math

class charbonnier_loss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-8):
        super(charbonnier_loss, self).__init__()
        self.eps = eps


    def forward(self, pred, gt):
        diff = pred - gt
        charbonnier_loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return charbonnier_loss

class perceptual_loss(nn.Module):

    def __init__(self):
        super(perceptual_loss, self).__init__()

    def forward(self,pred,gt):
            vgg = vgg16(pretrained=True)
            vgg = vgg.cuda()
            loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            
            mse_loss = nn.MSELoss()
            
            perception_loss = mse_loss(loss_network(pred), loss_network(gt))
            
            return perception_loss

class bicubic_loss(nn.Module):

    def __init__(self):
        super(bicubic_loss, self).__init__()


    def forward(self, gt, hr_pred):
        t,c,h,w = gt.size()
        lr = F.interpolate(gt,size=(gt.size(2)//4,gt.size(3)//4),mode='bicubic',align_corners=True)
        lr_pred = F.interpolate(hr_pred,size=(hr_pred.size(2)//4,hr_pred.size(3)//4),mode='bicubic',align_corners=True)
        diff = lr_pred - lr
        bicubic_loss = torch.sum(torch.sqrt(diff * diff))
        return bicubic_loss
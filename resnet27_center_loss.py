#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:28:41 2018

@author: junyang
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(dim_in,dim_out,3,1,1,bias=False),#Bx(3)64x224x224->Bx64x224x224                
                nn.PReLU(),
                nn.Conv2d(dim_out,dim_out,3,1,1,bias=False),#Bx64x224x224->Bx64x224x224
                nn.PReLU(),                              
                )

    def forward(self,input):
        x = input + self.main(input)
        return x

class downsample(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(downsample, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(dim_in,dim_out,kernel_size=3, stride=1, padding=0,bias=False),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )
    def forward(self,input):
        x = self.main(input)
        return x

class Resnet27(nn.Module):
    def __init__(self):
        super(Resnet27,self).__init__()
        #self.features = None
        self.conv1 = nn.Sequential(
                nn.Conv2d(3,32,kernel_size=3, stride=1, padding=0,bias=False),
                nn.PReLU()
                )
        sequence = []
        sequence += [downsample(32,64)]
        sequence += [ResidualBlock(64,64)]
        sequence += [downsample(64,128)]
        for i in range(2):
            sequence += [ResidualBlock(128,128)]
        sequence += [downsample(128,256)]
        for i in range(5):
            sequence += [ResidualBlock(256,256)]
        sequence += [downsample(256,512)]
        for i in range(3):
            sequence += [ResidualBlock(512,512)]
        self.downsample_ResidualBlock = nn.Sequential(*sequence)
        #self.avg_pool = nn.AvgPool2d(7,stride=1)
        self.classifier1 = nn.Linear(512*7*7,512)
        self.classifier2 = nn.Linear(512,333)
        
    def forward(self,input):
        x = self.conv1(input)
        x = self.downsample_ResidualBlock(x)
        #x = self.avg_pool(x)
        #x = x.squeeze()
        x = x.view(x.size(0),-1)
        x = self.classifier1(x)
        features = x
        x = self.classifier2(x)
        return features,x

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        #print(x.size())
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        distmat = distmat + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:28:41 2018

@author: junyang
"""

import torch
import torch.nn as nn

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
        self.features = None
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
        self.avg_pool = nn.AvgPool2d(7,stride=1)
        self.classifier = nn.Linear(512,333)
        
    def forward(self,input):
        x = self.conv1(input)
        x = self.downsample_ResidualBlock(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        self.features = x
        x = self.classifier(x)
        return x
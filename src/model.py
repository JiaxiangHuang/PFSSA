# -*- coding: utf-8 -*-
"""
@File ：model.py
@Auth ：Jiaxiang Huang
@Time ：11/20/23 4:17 PM
"""
from __future__ import print_function
import torch
import torch.nn as nn


class ASC3d(nn.Module):
    def __init__(self):
        super(ASC3d, self).__init__()

    def forward(self, input):
        s = torch.sum(input, dim=1)
        suq= s.unsqueeze(1)
        constrained = input / suq
        return constrained


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PCAlayer(nn.Module):
    def __init__(self):
        super(PCAlayer, self).__init__()
    def forward(self, X, k, center=True):
        """
        param X: BxCxHxW
        param k: scalar
        return:
        """
        B, C, H, W = X.shape
        X = X.permute(0, 2, 3, 1)  # BxHxWxC
        X = X.reshape(B, H * W, C)
        U, S, V = torch.pca_lowrank(X, k, center=center)
        Y = torch.bmm(X, V[:, :, :k])
        Y = Y.reshape(B, H, W, k)
        Y = Y.permute(0, 3, 1, 2)  # BxHxWxk
        return Y

# --------------model----------------------#
class PFSSA(nn.Module):
    def __init__(self, num_bands, threshold, end_m, planes, pca_out_dim):
        super(PFSSA, self).__init__()
        self.pca_out_dim = pca_out_dim
        self.pca = PCAlayer()

        self.conv0 = nn.Conv2d(num_bands, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act0 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=( 1, 1), padding=( 1, 1))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act3 = nn.ReLU()

        self.transpose1 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.transpose3 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.transpose4 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv4 = nn.Conv2d(64, end_m, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.soft = nn.Softplus(threshold=threshold)
        self.asc = ASC3d()

        self.ca = ChannelAttention(planes*4)
        self.sa = SpatialAttention()
        self.ca1 = ChannelAttention(planes )
        self.ca2 = ChannelAttention(planes * 2)
        self.ca3 = ChannelAttention(planes * 4)
        self.ca_trans1 = ChannelAttention(planes* 2)
        self.ca_trans2 = ChannelAttention(planes)

    def forward(self, img):

        out = self.conv0(img)
        x0 = self.act0(out)

        out = self.conv1(x0)
        x1 = self.act1(out)
        x1_pool1 = self.pool1(x1)

        out = self.conv2(x1_pool1)
        x2 = self.act2(out)
        x2_pool2 = self.pool2(x2)

        out = self.conv3(x2_pool2)
        x3 = self.act3(out)

        x4 = self.transpose1(x3)

        x5 = self.transpose2(x2 + x4 )

        x5 = self.ca_trans2(x5) * x5
        x5 = self.sa(x5) * x5

        x6 = self.conv4(x5 + x1)

        x7 = self.soft(x6)

        out = self.asc(x7)
        return out

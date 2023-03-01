# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 11:38
# @Author  : JunL
# @File    : decode.py
# @Software: PyCharm 
# @Comment :解码阶段：使用4组(3*3*3卷积和转置卷积)还原到原来的shape
# @Comment :使用1*1*1返回分类通道数
import paddle.nn as nn


class UpsamplingDeconvBlock(nn.layer):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3DTranspose(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3D(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3D(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3DTranspose(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


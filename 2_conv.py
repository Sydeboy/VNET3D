# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 20:31
# @Author  : JunL
# @File    : 2_conv.py
# @Software: PyCharm 
# @Comment :编码阶段采用四组（3*3*3 + 2*2*2）卷积
import paddle.nn as nn


class DownsamplingConvBlock(nn.layer):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3D(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3D(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3D(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3D(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


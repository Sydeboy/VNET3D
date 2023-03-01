# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 19:01
# @Author  : JunL
# @File    : 3_conv.py
# @Software: PyCharm 
# @Comment :编码阶段采用四组（3*3*3 + 2*2*2）卷积
import paddle.nn as nn


class ConvBlock(nn.layer):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3D(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3D(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3D(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU())

        self.conv = nn.Sequential(*ops)  # 整合


def forward(self, x):
    x = self.conv(x)
    return x

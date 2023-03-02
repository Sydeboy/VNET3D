# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 16:11
# @Author  : JunL
# @File    : vnet_multi_head.py
# @Software: PyCharm 
# @Comment :vnet网络主体架构
import paddle.nn as nn
import paddle


class ConvBlock(nn.layer):
    # torch版本 class ConvBlock(nn.Module)
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        # ConvBlock继承父类
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
            elif normalization == 'none':
                assert False
            ops.append(nn.ReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.layer):
    """
    残差+卷积
    """
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

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
            elif normalization == 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU())

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

class DownsamplingConvBlock(nn.layer):


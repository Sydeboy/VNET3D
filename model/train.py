# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 20:16
# @Author  : JunL
# @File    : train.py
# @Software: PyCharm 
# @Comment :训练脚本
import argparse
import logging
import os
import numpy as np
import paddle
import paddle.vision import transforms
from scipy.ndimage import distance_transform_edt as distance
from vnet_multi_head import VNetMultiHead
from data_fix import LA

"""
Train a multi-head vnet to output 
1) predicted segmentation
2) regress the distance transform map 
e.g.
Deep Distance Transform for Tubular Structure Segmentation in CT Scans
https://arxiv.org/abs/1912.03383
Shape-Aware Complementary-Task Learning for Multi-Organ Segmentation
https://arxiv.org/abs/1908.05099
"""
parser = argparse.ArgumentParser()
# 注意修改这里的路径，暂时不改
parser.add_argument('--root_path', type=str, default='./C4/C4_Z=20', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet_dp_la_MH_FGDTM_L1PlusL2', help='model_name;dp:add dropout; MH:multi-head')
parser.add_argument('--max_iterations', type=int,  default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model_la/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

patch_size = (112, 112, 80)
num_classes = 2

def dice_loss(score, target):
    """
    dice损失函数 target = target.float()
    :param score:
    :param target:
    :return:
    """
    target = paddle.cast(target, 'float32')
    smooth = 1e-5
    intersect = paddle.sum(score * target)
    y_sum = paddle.sum(target * target)
    z_sum = paddle.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1- loss
    return loss

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    计算二进制掩码中前景的距离变换图
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) # 前景距离图
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    :param img_gt:
    :param out_shape:
    :return:
    """
    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


if __name__ == "__main__":

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='&H:&M:&S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNetMultiHead(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)

    db_train =



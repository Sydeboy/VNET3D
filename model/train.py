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
import time
import random
import shutil  # os模块的补充， 提供了复制移动删除等操作
import sys
import paddle.nn.functional as F

from tqdm import tqdm  # 进度条
from paddle.vision import transforms
from paddle.io import DataLoader
from scipy.ndimage import distance_transform_edt as distance
from vnet_multi_head import VNetMultiHead
from data.data_fix import LAHeart, RandomCrop, RandomRotFlip, ToTensor

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
parser.add_argument('--exp', type=str, default='vnet_dp_la_MH_FGDTM_L1PlusL2',
                    help='model_name;dp:add dropout; MH:multi-head')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model_la/" + args.exp + "/"  # 拼接路径，exp的路径

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 指定使用的显卡
batch_size = args.batch_size * len(args.gpu.split(','))  # 多GPU训练
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
    target = paddle.cast(target, 'float32')  # 转换类型
    smooth = 1e-5
    intersect = paddle.sum(score * target)
    y_sum = paddle.sum(target * target)
    z_sum = paddle.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
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

    db_train = LAHeart(base_dir=train_data_path, split='train', num=16,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(),
                           ToTensor(),
                       ]))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)

    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=base_lr, momentum=0.9,
                                          weight_decay=0.0001)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    for epoch in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch[0], sampled_batch[1]

            outputs, out_dis = net(volume_batch)

            with paddle.no_grad():
                gt_dis = compute_dtm(label_batch.cpu().numpy(), out_dis.shape)
                gt_dis = paddle.to_tensor(gt_dis, place=paddle.CUDAPlace(0), dtype='float32')

            outputs = paddle.transpose(outputs, perm=[0, 2, 3, 4, 1])
            # compute CE + Dice loss
            # loss_ce = F.cross_entropy(outputs, label_batch)
            loss_ce = F.cross_entropy(input=outputs, label=label_batch)

            outputs = paddle.transpose(outputs, perm=[0, 4, 1, 2, 3])
            outputs_soft = F.softmax(outputs, axis=1)

            loss_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            # compute L1 + L2 Loss
            loss_dist = paddle.norm(out_dis - gt_dis, p=1) / paddle.numel(out_dis) + F.mse_loss(out_dis, gt_dis)

            loss = loss_ce + loss_dice + loss_dist

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            iter_num = iter_num + 1
            logging.info('iteration %d : loss_dist : %f' % (iter_num, loss_dist.numpy()))
            logging.info('iteration %d : loss_dice : %f' % (iter_num, loss_dice.numpy()))
            logging.info('iteration %d : loss : %f' % (iter_num, loss.numpy()))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 1000)
                optimizer.set_lr(lr_)

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pdparams')
                paddle.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pdparams')
        paddle.save(net.state_dict(), save_mode_path)
        paddle.save(optimizer.state_dict(), save_mode_path.replace('.pdparams', '.pdopt'))
        logging.info("save model to {}".format(save_mode_path))

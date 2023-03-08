# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:35
# @Author  : JunL
# @File    : test.py
# @Software: PyCharm 
# @Comment :测试脚本
import os
import argparse
from vnet_multi_head import VNetMultiHead
import h5py
import math
import nibabel as nib  # 用于读取和写入神经影像数据格式（如 NIfTI 格式）的工具
import numpy as np
from medpy import metric
import paddle.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict
import paddle

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../C4/C4_Z=20/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='../../LA_data_h5/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str, default='vnet_dp_la_MH_FGDTM_L1PlusL2', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch_num', type=int, default='6000', help='checkpoint to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model_la/" + FLAGS.model + "/"
test_save_path = "../model_la/prediction/" + FLAGS.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

# 定义文件路径
test_path = r'E:\code\VNET\C4\test.list'
with open(test_path, 'r') as f:
    image_list = f.readlines()
# with open(FLAGS.root_path + '../test.list', 'r') as f:
#     image_list = f.readlines()
image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNetMultiHead(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False)
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pdparams')
    net.load_dict(paddle.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = dist_test_all_case(net, image_list, num_classes=num_classes,
                                    patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                    save_result=True, test_save_path=test_save_path)

    return avg_metric


def dist_test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                       save_result=True, test_save_path=None, preproc_fn=None):
    """
    :param net: 神经网络模型
    :param image_list:测试的图像路径
    :param num_classes:类别数
    :param patch_size:滑动窗口大小
    :param stride_xy:步幅
    :param stride_z:
    :param save_result:保存结果
    :param test_save_path:保存路径
    :param preproc_fn:预处理图像的函数
    :return:
    """
    total_metric = 0.0
    metric_dict = OrderedDict()  # 存储每个图像的分割指标值
    metric_dict['name'] = list()
    metric_dict['dice'] = list()  # Dice系数
    metric_dict['jaccard'] = list()  # IOU系数
    metric_dict['asd'] = list()  # 平均表面距离ASD
    metric_dict['95hd'] = list()  # Hausdorff距离
    for image_path in tqdm(image_list):
        case_name = image_path.split('/')[-2]
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')  # 读取HDF5格式的图像文件
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)

        # 调用该函数对当前图像进行推理，得到预测分割，得分图和距离图
        prediction, score_map, pred_dist = test_single_case(net, image, stride_xy, stride_z, patch_size,
                                                            num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            print(single_metric)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(single_metric[0])
            metric_dict['jaccard'].append(single_metric[1])
            metric_dict['asd'].append(single_metric[2])
            metric_dict['95hd'].append(single_metric[3])

        total_metric += np.asarray(single_metric)

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            # 1.预测结果
            # 2.测试数据本身
            # 3.数据的真实标签
            # 4.概率分布
            # np.eye(4)创建单位矩阵
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path_temp + '/' + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                     test_save_path_temp + '/' + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                     test_save_path_temp + '/' + id + "_gt.nii.gz")
            nib.save(nib.Nifti1Image(pred_dist[:].astype(np.float32), np.eye(4)),
                     test_save_path_temp + '/' + id + "_dist.nii.gz")
    avg_metric = total_metric / len(image_list)
    # pd.DataFrame()用于创建表格形式的数据结构
    metric_csv = pd.DataFrame(metric_dict)
    metric_csv.to_csv(test_save_path + '/metric_' + str(FLAGS.epoch_num) + '.csv', index=False)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    # padding填充
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    # math.ceil()表示向上取整，以确保覆盖整个图像
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    pred_dist = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = paddle.to_tensor(data=test_patch, place=paddle.CUDAPlace(0))
                y1, out_dist = net(test_patch)
                y = F.softmax(y1, axis=1)
                y = y.cpu().numpy()
                y = y[0, :, :, :, :]
                out_dist = out_dist.cpu().numpy()
                pred_dist[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = pred_dist[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + out_dist[0, 0, :,
                                                                                                       :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    pred_dist = pred_dist / cnt
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        pred_dist = pred_dist[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map, pred_dist


def cal_dice(prediction, label, num=2):
    """
    :param prediction: 预测结果-->numpy数组
    :param label: 标签-->numpy数组
    :param num:类别数
    :return:Dice系数
    """
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    """
    二进制binary
    metric.binary.dc是评价指标库，计算Dice系数
    :param pred:预测值
    :param gt: 真实值
    :return: 各个指标
    """
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.epoch_num)
    # print(metric)

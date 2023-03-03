# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 11:55
# @Author  : JunL
# @File    : data_fix.py
# @Software: PyCharm 
# @Comment :数据集划分-->随机裁剪-->随机翻转-->随机噪声-->one-hot编码-->转换张量

import os
import numpy as np
from glob import glob
import h5py
import itertools
from paddle.vision import transforms
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader


class LAHeart(Dataset):
    """ LA Dataset"""

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir + './../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + './../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # 读取
        # 首先拼接HDF5的文件路径，标记只读
        h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
        # 读取image的数据集，赋值到image，并读取全部
        image = h5f['image'][:]
        label = h5f['label'][:]
        # 创建sample字典，分别读取image和label
        sample = {'image': image, 'label': label}
        # 如果有transform则将sample作为输入
        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """
    随机裁剪
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size(0))
        h1 = np.random.randint(0, h - self.output_size(1))
        d1 = np.random.randint(0, d - self.output_size(2))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    随机翻转
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    """
    添加随机噪声
    """

    def __init__(self, mu=0, sigma=0.1):
        # 噪声的均值和标准差
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 1.先生成与输入图像一致大小的随机噪声矩阵，并乘以sigma -->得到给定均值和标准差的高斯分布噪声
        # 2.np.clip()将噪声先知道-2sigma到2sigma之间
        noise = np.clip(self.sigma * np.random.randint(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu  # 可以改写为 noise += self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    """
    func:将给定的标签数据进行one-hot编码，将每个可能的标签值编码为一个向量
         其中只有与该标签值对应的位置为1，其余位置为0
    """

    def __init__(self, num_class):
        # num_class表示创建one-hot编码类别数,也就是标签类别数
        self.num_class = num_class

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 1.生成一个全零矩阵--> [num_class, h, w, d]
        # 2.对每个标签类别，将one-hot中对应的通道，也就是i通道中与之对应相同的像素点值设为1
        onehot_label = np.zeros((self.num_class, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_class):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """
    convert ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # 将样本数据转化到PaddlePaddle张量
        if 'onehot_label' in sample:
            # torch版本 return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
            # 'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
            return paddle.to_tensor(image, dtype='float32'), paddle.to_tensor(sample['label'],
                                                                              dtype='int64'), paddle.to_tensor(
                sample['onehot_label'], dtype='int64')
        # tips:默认在CPU上训练，如果需要gpu上计算则需要使用paddle.to_device方法
        else:
            # torch版本
            # return {'image': torch.from_numpy(image), xxx(sample['label']).long()}
            # long()确保张量和索引操作兼容，确保生成的张量具有预期使用的正确数据类型
            return paddle.to_tensor(image, dtype='float32'), paddle.to_tensor(sample['label'], dtype='int64')


def iterate_one(iterable):
    """
    接收一个可迭代的对象。只迭代一次
    :param iterable:
    :return: 返回随机排列的迭代器
    """
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    """
    np.random.permutation()用于对数据集进行随机排列，打乱数据集顺序。迭代多次
    :param indices: 索引列表
    :return: 无限循环的迭代器用于迭代整个数据集
    """

    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    将一个可迭代对象iterable拆分成固定长度的块
    grouper('ABCDEFG', 3) --> ABC DEF
    :param iterable: 可迭代对象
    :param n: 每个块的长度
    :return:
    """
    args = [iter(iterable)] * n  # 将iterable重复n次放入列表中
    # 将重复n次的迭代器对象组合成一个新的迭代器对象
    # 由于zip()函数会返回一个元组的迭代器，其中元组的每个元素都是来自于各个迭代器中的对应位置的元素，因此这里使用 *args 对列表中的元素进行解包。
    return zip(*args)

# DB = LAHeart(base_dir='../../C4/C4_Z=20',num=16,
#                        transform = transforms.Compose([
#                           RandomRotFlip(),
#                           RandomCrop((112, 112, 80)),
#                           ToTensor(),
#                           ]))
#
# trainloader = DataLoader(DB, batch_size=8, shuffle=True)
# #在使用 trainloader 对数据进行迭代时，可以得到一个批次的数据，其中每个样本包括一个 3D 图像数据
# 和一个相应的标签。在这个示例中，使用一个 for 循环来迭代训练集中的所有批次数据。在每次迭代中，将一个批次的数据打印出来。
# for i_batch, sampled_batch in enumerate(trainloader()):
#     # generate paired iput
#     print(sampled_batch)
# volume_batch, label_batch = sampled_batch[0], sampled_batch[1]
# print('volume_batch.shape:{}, label_batch.shape:{}'.format(volume_batch.shape, label_batch.shape))

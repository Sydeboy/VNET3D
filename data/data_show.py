# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 11:48
# @Author  : JunL
# @File    : data_show.py
# @Software: PyCharm 
# @Comment :展示3D的咽癌图片
from matplotlib import pyplot as plt
import SimpleITK as sitk  # 医学图像经常使用的库
import numpy as np

img_path = '4C2021_C04_TLS01/Train Set 01/001.nii.gz'
label_path = '4C2021_C04_TLS01/Train Label 01/001.nii.gz'


# 展示选定的切片图片
I = sitk.ReadImage(img_path)  # 读取图像，如.mhd, .nii, .nrrd
img = sitk.GetArrayFromImage(I)  # 将sitk图像转换为数组
print(type(img))  # <class 'numpy.ndarray'>
print(img.shape)  # (95, 512, 512)
# plt.imshow(img[66, ...], cmap='gray', interpolation='bicubic')
plt.imshow(img[66, ...], cmap='gray', interpolation='bicubic')
plt.show()

# 展示标签
I = sitk.ReadImage(label_path)
label = sitk.GetArrayFromImage(I)
print(type(I))
print(label.shape)
plt.imshow(label[66, ...], cmap='gray', interpolation='bicubic')
plt.show()

# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:00
# @Author  : JunL
# @File    : test_softmax.py
# @Software: PyCharm 
# @Comment :测试softmax
import paddle
import paddle.nn.functional as F

num = 4 * 2 * 6 * 6 * 5
out1 = paddle.arange(num, dtype='float32')  # 一维张量
# print(out1)
out = paddle.reshape(out1, [4, 2, 6, 6, 5])  # 五维
# print(out[0])
# print(out[:, 1:2, :, :, :])
# 可以理解为4 * [2, 6, 6, 5] --> 4*2*[6, 6, 5] -->4*2*6[6, 5]
# print(out)
outputs_soft = F.softmax(out, axis=1)  # 沿第二个轴进行softmax
print(outputs_soft)

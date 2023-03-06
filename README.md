# VNET3D
3d Vnet learning



## Dice Loss

### 1.Dice系数与IOU

- 解决类别不平衡的问题
- 来源Dice系数，一个用来衡量两个样本相似度的度量函数
- 衡量两个集合的交并比


#### latex形式
![](http://latex.codecogs.com/svg.latex?公式代码)

$$
Dice=\frac{2|{X}\bigcap{Y}|}{|X|+|Y|}=\frac{2TP}{2TP+FP+TN} \tag{1}
$$

Iou又叫做Jaccard系数：

$$
Jaccard=\frac{|{X}\bigcap{Y}|}{|X\bigcup{Y}|}=\frac{TP}{TP+FP+TN}\tag{2}
$$

### 2.Dice Loss

在V-Net中，Dice损失定义为1-Dice系数。在分割算法中，假设像素点总数为$N$,$p_i$表示第$i$个样本的问题，$g_i$表示该像素的Ground Truth，那么Dice Loss可以表示为：
![](http://latex.codecogs.com/svg.latex?公式代码)
$$
\ell_{dice}=\frac{2\sum{_i}{}
$$

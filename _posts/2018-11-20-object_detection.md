---
layout:     post   				    # 使用的布局
title:      46.0 目标检测			# 标题 
date:       2018-11-20 				# 时间
author:     子颢 						# 作者
catalog: true 						# 是否归档
tags:								# 标签
    - 计算机视觉
    - 目标检测
---

目标检测（Object Detection），就是在给定的图片中精确找到物体所在位置（方框框出它的位置），并标注出物体的类别。object detection技术的演进：RCNN->Fast RCNN->Faster RCNN。
![OD](/img/OD-01.png)

# RCNN

RCNN（Region CNN）,是最古老也是最经典的基于深度学习的目标检测算法，该算法使用Selective Search方法预先从图片（训练数据）中提取一系列较可能是物体的候选区域，之后仅在这些候选区域上提取特征，进行判断。RCNN算法主要分为4个步骤：
1. 从训练数据图像中生成1K~2K个候选区域（region proposals）。
2. 对每个候选区域，使用CNN提取特征。
3. 将提取到的特征送入SVM分类器，判断所属类别。
4. 使用回归器精细修正候选框位置。
![OD](/img/OD-02.png)
注：训练数据一共包含10000张图像，一共20个类别，每张图片中，标注物体的类别和位置。
下面我们依次来看一看每一步具体是怎么做的。

## 候选区域生成

候选区域生成使用了Selective Search方法从一张图像中生成1000~2000个候选区域，基本思路如下：
1. 使用一种过分割手段，将图像分割成一个个小region。
2. 采用类似于层次聚类的方法依次合并吻合度最高的两个region。
3. 输出聚合后的所有region（即得到RoI，Region of Interests）。
注：候选区域生成步骤相对独立，可以使用任意算法进行，只不过Selective Search算法最为常用。

## 特征提取

1. 预处理。提取特征之前，首先把所有候选区域归一化成同一尺寸，比如227×227。
2. 采用Image Net或经典的ResNet训练分类模型。训练数据一共有21个label（20个物体label加1个背景label），考察每一个候选区域和当前图像上的所有标注框的重叠面积，如果重叠比例大于0.5，则认为此候选区域为此标注的类别，否则认为此候选区域为背景。
3. 对每个候选区域，输入到2步中的CNN网络中，取倒数第二层的输出（4096维）作为特征表示。

## 类别判断

1. 对每一个类别，训练一个线性SVM二分类器。
2. 输入为特征提取步所输出的4096维特征表示，输出是否属于此类（0或1）。
3. 对于每一个类别（对应每一个不同的SVM二分类器），考察每一个候选区域，如果和本类所有标注框的重叠都小于0.3，认定其为负样本，取本类的标注区域作为正样本。
4. 由于负样本很多，使用hard negative mining方法进行下采样。

注：这里为什么要把特征提取和类别判断拆成两步来做？原因有二：
1. 因为svm训练和cnn训练过程的正负样本定义方式各有不同，导致最后采用CNN softmax输出比采用svm精度还低。事情是这样的，cnn在训练的时候，对训练数据做了比较宽松的标注，比如一个bounding box可能只包含物体的一部分，那么我也把它标注为正样本，用于训练cnn。采用这个方法的主要原因在于因为CNN容易过拟合，所以需要大量的训练数据，所以在CNN训练阶段我们是对Bounding box的位置限制条件限制的比较松(IOU只要大于0.5都被标注为正样本了)。然而svm训练的时候，因为svm适用于少样本训练，所以对于训练样本数据的IOU要求比较严格，我们只有当bounding box把整个物体都包含进去了，我们才把它标注为物体类别，然后训练svm。
2. 这也相当于做了一次stack ensemble。

## 位置精修

训练一个线性回归模型精细修正候选框位置。训练样本为所有和真值标注重叠面积大于0.6的候选区域，输入为特征提取步所得到的4096维特征表示，输出为xy方向上的缩放和平移（一维向量得到一个回归值，两列矩阵即可得到两个回归值）。

# Fast RCNN

Fast RCNN顾名思义，就是在RCNN的基础上进行改进，其构思更加精巧，流程更为紧凑，大幅提升了目标检测的速度。Fast RCNN和RCNN相比，训练时间从84小时减少为9.5小时，测试时间从47秒减少为0.32秒，效果也相对有所提升。Fast RCNN主要从以下几个方面对RCNN做了改进，下面我们分别来看。
![OD](/img/OD-03.png)

## 空间金字塔池化

空间金字塔池化（SPP，spatial pyramid pooling），他的作用就是可以让网络输入任意大小的图片（region proposal），但是会生成固定大小的输出。我们知道，CNN一般都含有卷积部分和全连接部分，其中卷积层不需要固定尺寸的图像输入，而全连接层是需要固定大小的输入的，所以在RCNN中，需要将每个Region Proposal缩放或裁剪（warp）成统一的227x227的大小并输入到CNN，但是这种简单粗暴的预处理，会导致图像要么被拉伸变形、要么物体不全，因此限制了识别精度。SPP刚好解决了这个问题，他不需要对图像进行crop和wrap，而是在卷积层的最后加入了spatial pyramid pooling layer：
![OD](/img/OD-04.png)
使得网络的输入图像可以是任意尺寸的，输出则是一个固定维数的向量，那么SPP具体是怎么做到的呢？原理非常简单，即在卷积之后的feature map上（下图黑色图片表示），用三张大小不同的网格（分别是4x4，2x2，1x1）扣在这张特征图上，就可以得到16+4+1=21种不同的块（Spatial bins），我们分别从每个块中提取出一个特征，这样刚好就是我们要提取的21维特征向量，这种以不同大小的格子的组合方式来池化的过程就是空间金字塔池化（SPP）。比如，要进行空间金字塔最大池化，其实就是从这21个图片块中，分别计算每个块的最大值，从而得到一个输出单元，最终得到一个21维特征的输出。所以Conv5计算出的feature map也是任意大小的，现在经过SPP之后，就可以变成固定大小的输出了，以下图为例，一共可以输出（16+4+1）x256 = 5376维的特征。
![OD](/img/OD-05.png)

## 一次特征提取

RCNN的第一步中对原始图片通过Selective Search提取的region proposal（候选区域、候选框）多达2000个，每个候选框都需要进行CNN提特征+SVM分类，计算量很大，这是导致RCNN速度慢的主要原因之一。考虑这2000个region proposal，其实都是图像的一部分，那么我们完全可以对图像提一次卷积层特征，得到feature maps，然后将region proposal在原图的位置映射到卷积层特征图上，再对各个候选框在特征图上的映射结果采用金字塔空间池化，提取出固定长度的特征向量，并以此作为每个region proposal的特征表示，这就是Fast RCNN的做法。

## multi-task

Fast RCNN采用多任务迁移学习模型，将边框回归Bounding Box Regression和Object Classification加在一起训练multi-task model。Fast RCNN真正将RCNN的类别判断和位置精修步合二为一，直接使用softmax替代SVM进行多分类，同时将边框回归也加入进行端到端联合训练。
![OD](/img/OD-06.png)

# Faster RCNN

Faster RCNN，顾名思义，就是对Fast RCNN的改进，在Fast RCNN的基础上通过引入Region Proposal Network（RPN）替代Selective Search来提取的候选区域。
![OD](/img/OD-07.png)

## RPN

1. 原始图像经过Fast RCNN的一次特征步骤，得到51x39x256维的feature maps，可以将之看做是一张51x39的256通道“图像”，对于该图像的每一个“像素”位置，考虑9个候选窗口（anchor）：三种面积{128^2,256^2,512^2} X 三种比例{1:1,1:2,2:1}。如下所示：
![OD](/img/OD-08.png)
2. 9个anchor就是9个滑动窗口，分别在feature maps上以“像素”为单位滑动，作为Region Proposal的候选（Proposal of the Region Proposal）。
3. 将每个anchor得到的候选输入一个multi-task model，该model包括两个任务，一个分类任务和一个回归任务。分类任务输出在每一个“像素”位置上，9个anchor属于物体和背景的概率（二分类）；回归任务输出每一个“像素”位置上，9个anchor对应窗口应该平移缩放的参数。（就局部来说，这是一个全连接神经网络；就全局来说，由于网络在所有51x39个位置的参数相同，所以实际上是用1×1的卷积网络实现）

## 联合训练

1. 将RPN得到的所有Region Proposal，送入Fast RCNN的multi-task模型进行再训练。所以Faster RCNN一共包含4个损失函数：RPN calssification(anchor good.bad)、RPN regression(anchor -> propoasal)、Fast R-CNN classification(over classes)、Fast R-CNN regression(proposal -> box)。
2. 训练时可以交替进行，即先训练RPN若干次，提取候选区域，然后固定所有参数（包括一次特征提取部分），用候选区域训练Fast RCNN，然后固定Fast RCNN参数，训练RPN，交替往复迭代训练，直到收敛。

# 社群

- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
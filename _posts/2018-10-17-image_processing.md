---
layout:     post   				    # 使用的布局
title:      41.0 图像处理				# 标题 
date:       2018-10-17 				# 时间
author:     子颢 						# 作者
catalog: true 						# 是否归档
tags:								#标签
    - 计算机视觉
    - computer vision
    - CV
    - 图像处理
    - opencv
    - PIL
    - skimage
---

计算机视觉（computer vision，CV），是不同于NLP的一个全新的领域，主要有图像识别、图像分割、目标检测、语义分割等几个不同的方向。其实CV和NLP的底层原理都是相通的，都使用多层神经网络结构，只是输入层的不同，NLP需要先将自然语言做word embedding后才能输入给模型，而CV的输入可以直接是图像的像素点的像素值。正因为如此，CV在他的不同方向上都有比较成熟的应用和技术，而NLP相对来说并没有发展的那么成熟，语义理解一直是NLP的难点和瓶颈。

我们一步步来，在对图像进行深度分析之前，首先需要对图像进行预处理，下面我们就来看下图像处理的软件和工具都有哪些？都有哪些处理方法呢？其实主流的Python图像处理库包括以下几种：opencv、PIL、matplotlib、scipy.misc、skimage。

# opencv

opencv是目前最常用的图像处理库，没有之一，功能全面且非常强大。安装方式如下：
```
pip3 install opencv-python
```

## 图像读取

```
import cv2

# 读入图片：默认为彩色图（cv2.IMREAD_COLOR），灰度图（cv2.IMREAD_GRAYSCALE）
# img_array = cv2.imread('data/test.png', cv2.IMREAD_GRAYSCALE)
img_array = cv2.imread('data/test.png')
# 也可以先读入彩色图，再转灰度图
# img_array = cv2.imread('data/test.png')
# img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

# 图片展示：src为展示窗口name，img_array为数组
cv2.imshow('src', img_array)
print(img_array)
print(img_array.shape)  # 如果以彩色图读入，shape为(h,w,c)；如果以灰度图读入，shape为(h,w)
print(img_array.size)  # 像素总数目
print(img_array.dtype)  # 数组类型
cv2.waitKey()  # 线程wait
```
运行结果：
```
[[[ 95  75  73]
  [ 84  57  54]
  [ 84  57  54]
  ...
]]
(740, 1352, 3)
3001440
uint8
```
注意：
1. 调用cv2.imread读进来的图片已经是一个numpy矩阵了，矩阵元素就是像素点的值（0~255内的正整数）。
2. opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB，可以通过cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)转为RGB排列。

## 图像归一化

在对图像进行卷积等操作前，一般需要归一化。
```
# img_array = img_array.astype("float") / 255.0  # 这一步可省略
img_array = img_array / 255.0
print(img_array.dtype)
print(img_array)
```
运行结果：
```
float64
[[[0.37254902 0.29411765 0.28627451]
  [0.32941176 0.22352941 0.21176471]
  [0.32941176 0.22352941 0.21176471]
  ...
 ]]
```

## 图像存储

```
cv2.imwrite('data/test1.jpg', img_array)  # 归一化后得到的是全黑的图片
cv2.imwrite('data/test2.jpg', img_array * 255)  # 这样就还原了原图片
```

计算机视觉中有一个专有名词叫ROI（region of interest），感兴趣区域，即从图像中选择一个区域，这个区域是你做图像分析所关注的重点，圈定该区域以便进行进一步处理。使用ROI圈定你想读的目标，可以减少处理时间，增加精度。

## roi

```
roi = img_array[200:550, 100:450, :]
cv2.imshow('roi', roi)

b, g, r = cv2.split(img_array)  # 通道拆分
img_array = cv2.merge((b, g, r))  # 通道合并
img_array[:, :, 2] = 0  # 将红色通道值全部设0
```

# PIL

PIL（Python Imaging Library），也即我们常称的Pillow，是一个很流行的图像库，它比opencv更为轻巧，正因如此，它深受大众的喜爱。
```
pip3 install Pillow
```
```
from PIL import Image
import numpy as np

img = Image.open('data/test.png')
print(img.format)  # PNG
print(img.size)  # 输出(w，h)，省略了c
print(img.mode)  # 输出RGBA。L为灰度图，RGB为真彩色，RGBA为加了透明通道
img.show()  # 显示图片

gray = img.convert('L')
print(gray.format)  # None
print(gray.size)  # 输出(w，h)
print(gray.mode)  # L
gray.show()

# pillow读进来的图片不是numpy array，而是一个对象，用下面方式将图片转numpy array
arr = np.array(img)
print(arr.shape)  # (h, w, c)，即(740, 1352, 4)
print(arr.dtype)  # uint8

# 图像存储
new_im = Image.fromarray(arr)
new_im.save('data/test_1.jpg')
```

# matplotlib

matplotlib是一个科学绘图神器，也是功能全面且强大的图像处理工具。
```
pip3 install matplotlib
```
```
import matplotlib.pyplot as plt

image = plt.imread('data/test.png')
plt.imshow(image)  # 优化图窗、坐标区和图像对象属性以便显示图像，相当于预热，调用show方法前必须先执行这一步动作
plt.show()  # 图像展示

# plt.imread读入后就是一个矩阵，跟opencv一样，但彩图的格式是RGB或RGBA，这是与opencv的区别
print(image.shape)  # (h,w,c)
print(image.dtype)  # float32
# [[[0.28627452 0.29411766 0.37254903 1.        ]
#   [0.21176471 0.22352941 0.32941177 1.        ]
#   [0.21176471 0.22352941 0.32941177 1.        ]
#   ...
# ]]]
print(image)
```
注意：读入后自动做了归一化处理。

# scipy.misc

SciPy是一个开源的Python算法库和数学工具包，他因为集成了各种丰富且优秀的算法模块而闻名，图像处理只是其中的一个模块。
```
pip3 install scipy
```
```
from scipy import misc

im = misc.imread('data/test.png')  # plt.imread读入后就是一个矩阵
# [[[ 73  75  95 255]
#   [ 54  57  84 255]
#   [ 54  57  84 255]
#   ...
# ]]]
print(im)
print(im.dtype)  # uint8
print(im.size)  # 4001920
print(im.shape)  # (740, 1352, 4)
misc.imsave('data/test0_1.png', im)  # 图像存储
```

# skimage

skimage，也即scikit-image，是基于scipy的一款图像处理包。
```
pip3 install scikit-image
```
```
from skimage import io

im = io.imread('data/test.png')
print(im)
print(im.shape)  # numpy矩阵，(h,w,c)
print(im.dtype)
print(im.size)
io.imshow(im)
io.imsave('data/sk.png',im)  # 图像存储
```

以上只是图像处理的一些基本用法，其实图像处理还包括很多复杂且有意义的操作，包括滤波、膨胀与腐蚀、边缘检测、霍夫变换、放射变换等，其实可以不用全部掌握，大可在需要用到的时候再具体深入，对症下药。想了解更多内容可以参看这篇博客：<a href="https://www.cnblogs.com/skyfsm/default.html?page=5" target="_blank">图像处理方法汇总与实战</a>

代码地址 <a href="https://github.com/qianshuang/CV" target="_blank">https://github.com/qianshuang/CV</a>

# 社群

- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
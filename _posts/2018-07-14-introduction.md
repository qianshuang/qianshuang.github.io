---
layout:     post   				    # 使用的布局（不需要改）
title:      00.0 AI系列分享概述及目录 				# 标题 
date:       2018-07-14 				# 时间
author:     子颢 						# 作者
# header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
---

# 概述

本系列分享的全部内容包括《机器学习》和《深度学习》两大部分，内容覆盖到机器学习和深度学习的方方面面，除了会循序渐进由易到难的介绍算法模型的原理和理论，主要还会介绍模型的实现和训练，以及模型的上线和调优。最重要的是会结合工业应用中的实际案例，让大家明白各种算法模型到底是如何在实际工作中落地和产出价值的。

我们会深入讲解机器学习深度学习在自然语言处理、计算机视觉、语音识别等三个领域的典型应用和落地方案。

如果你想投入AI的怀抱，但却苦于门槛太高无从下手，本系列分享将从零开始一步步带你入门和提高，为你指明学习的方向；如果你苦于缺乏AI实战经验，本系列分享其实更注重实践落地，每个案例教会你的不是解决某个问题，而是解决某类问题，使你能够触类旁通，举一反三。

# 系列分享大纲

## 机器学习部分

第一讲 算法预热
1. k近邻算法。案例：使用k近邻算法进行文本分类
2. 模型评测方法：准确率、召回率、混淆矩阵、ROC、AUC。

第二讲 概率论相关实战
1. 朴素贝叶斯实战
2. TF-IDF
3. 极大似然估计
4. 损失函数

第三讲 决策树及Ensemble方法（分类问题案例）
1. 决策树
2. 随机森林
3. AdaBoost
4. GBDT
5. Xgboost

第四讲 经典判别分类模型（分类问题案例）
1. 梯度下降、正则化
2. 逻辑回归（sklearn及TensorFlow实现）
3. softmax回归（sklearn及TensorFlow实现）
4. SVM（sklearn及TensorFlow实现）
5. 最大熵模型（TensorFlow实现）

第五讲 链式图模型
1. 隐马尔科夫模型
2. 条件随机场
3. 案例：CRF实现命名实体识别

第六讲 经典无监督算法及推荐系统实战
1. 期望最大化算法
2. k-means聚类
3. SimHash及以图搜图原理实现
4. PCA及其应用
5. 协同过滤、矩阵分解（FM）、主题模型（LDA）在推荐系统中的应用

## 深度学习部分

第七讲 深度学习基础
1. TensorFlow基础编程
2. 全连接神经网络（sklearn及TensorFlow实现基于词频统计的文本分类）
3. 神经网络精要（工业上神经网络训练技巧及经验总结）
4. TensorFlow高级特性（keras、eager exeution、wide & deep model）
5. word2vec & fasttext原理解析及实战

第八讲 CNN相关
1. TextCNN原理及实战
2. 案例：基于ContextCNN及HierarchicalCNN实现多轮对话意图分类
3. 案例：基于CNN及CRF实现命名实体识别
4. CNN在检索式问答系统中的应用（QQ-match）
5. CNN在语义检索中的应用（DSSM）

第九讲 RNN相关
1. RNN、LSTM、bi-LSTM、DRNN原理及实战
2. 案例：基于bi-LSTM及CRF实现命名实体识别
3. Attention机制（Global & local Attention、self Attention、multi-head Attention）
4. 案例：Seq2seq及Attention机制在生成式问答系统中的应用
5. 案例：Hierarchical Attention network在多轮对话意图分类中的应用
6. 基于Attention-CNN的语义检索模型

第十讲 知识图谱实战
1. 知识表示
2. 知识抽取
3. 知识融合
4. 知识存储
5. 知识推理
6. 实战案例1：基于elasticsearch的KBQA实现及示例
7. 实战案例2：基于REfO的KBQA实现及示例

第十一讲 迁移学习
1. 迁移学习在多领域场景下的应用。
2. 迁移学习在多语言场景下的应用。案例：多语言融合意图分类（生成对抗网络GAN）
3. 迁移学习在多任务场景下的应用。案例：分类与NER的muity-task实战

第十二讲 Autoencoders及推荐系统实战
1. Autoencoders在推荐系统中的应用
2. Wide & deep model在推荐系统中的应用

第十三讲 强化学习
1. 强化学习（Policy gradient、DQN、deep Q-learning）
2. 使用deep Q-learning强化学习算法玩吃豆人游戏

第十四讲 计算机视觉
1. 图像处理
2. OCR
3. OCR在身份证、发票等关键信息提取任务中的应用
4. 端到端OCR

第十五讲 计算机视觉进阶
1. 人脸识别
2. 目标检测
3. 图像语义分割
4. 视频分析

第十六讲 语音识别

第十七讲 语音合成

第十八讲 落地与总结
1. 机器学习模型部署及在线预测（PMML）
2. 深度学习模型部署及在线预测（TensorFlow Serving）
3. 总结以及给同学在工作中的意见和建议

# 社群
我们组建了技术社群，微信群里可以讨论AI相关的任何话题，分享以及活动内容会发布在微信公众号，感谢大家关注。
![562929489](/img/wx_qun.png)
![562929489](/img/wx_ewm.png)
---
layout:     post   				    # 使用的布局
title:      55.0 DeepFM	& DCN		# 标题 
date:       2020-07-27  			# 时间
author:     钱爽 						# 作者
catalog: true 						# 是否归档
tags:								# 标签
    - 推荐系统
---

FM(Factorization Machine，因子分解机)主要是为了解决数据稀疏的情况下，特征两两组合的问题。后人基于FM模型结合深度学习，进行了很多尝试，比如：
1. FNN（Factorization-machine supported Neural
Network），该模型先预训练FM，然后把得到的隐向量作为embedding的初始值，应用到DNN网络，因此该模型严重受限于FM的能力，并且FM的误差会级联传递下去。
2. PNN（Product-based Neural Network），在embedding层和MLP之间加入Product层，Product层就是将embedding后的特征向量两两内积（向量内积的结果是一个值）。也只能捕获两两特征之间的交互关系。
3. Wide & Deep model，之前讲过，wide部分仍然需要专家级的特征工程，才能知道应该把哪些特征之间进行cross product。

DeepFM（Factorization-Machine based neural network）模型能够端到端的学习all-order的特征交互，而不需要任何特征工程（一个特征叫1-order，两个特征cross叫2-order，n个特征cross叫n-order）。它通过将FM与DNN集成，FM负责model low-order特征交互，DNN负责model high-order特征交互。

# 模型架构

![DeepFM](/img/deepfm-01.png)
Addition其实代表concat。
![DeepFM](/img/deepfm-02.png)
FM部分与DNN部分共享同样的特征embedding，FM中的特征隐向量V同时作为了embedding向量。这样使得训练完成后的特征embedding能够同时准确的handle low and high-order特征交互。假设我们的embedding向量维度k=5，首先，对于输入的一条记录，同一个field只有一个位置是1，那么在由输入得到dense vector的过程中，输入层只有一个神经元起作用，得到的dense vector其实就是输入层到embedding层该神经元相连的五条线的权重，即vi1、vi2、vi3、vi4、vi5，这五个值组合起来就是我们在FM中所提到的Vi。在FM部分和DNN部分，这一块是共享权重的，对同一个特征来说，得到的Vi是相同的。

在FM中，特征i和j的交互是通过他们的隐向量计算内积完成。只要特征i或j出现在了训练数据中，FM就能够训练特征i或j的隐向量，然后特征i和j的交互直接通过隐向量内积得到。而对于之前的模型，想要学习特征i和j的交互，就必须特征i和j同时出现在同一个训练样本中，当数据稀疏时，这是不可能的。因此，对于训练数据中从未出现过的特征交互，FM也能很好的学习。

对于FM部分，模型表达式如下所示：
![DeepFM](/img/deepfm-03.png)
由此看出，模型的计算复杂度为O(n*n)，n为特征数量，为了简化计算，我们引入辅助向量V（维度为k），并将参数Wij改写为：
![DeepFM](/img/deepfm-04.png)
然后我们可以得到如下证明：
![DeepFM](/img/deepfm-05.png)
关于第一步的变换过程，证明如下：
![DeepFM](/img/deepfm-06.webp)
可以看出，变换后的计算复杂度为O(n)，大大简化了计算。

下面我们将结合代码进行讲解。

## 输入层
```
#feat_index是特征的一个序号，主要用于通过embedding_lookup选择我们的embedding。
feat_index = tf.placeholder(tf.int32, shape=[None,field_size], name='feat_index')
#feat_value是对应的特征值，如果是离散特征的话，就是1，如果不是离散特征的话，就保留原来的特征值。
feat_value = tf.placeholder(tf.float32, shape=[None,field_size], name='feat_value')
label = tf.placeholder(tf.float32,shape=[None,1], name='label')
```

## embedding层
```
#weights['feature_embeddings']矩阵中的每一行其实就是FM中的Vik，他的shape是f x k。f为所有特征one-hot后的总大小，K代表dense vector的维度。
weights['feature_embeddings'] = tf.Variable(tf.random_normal([feature_size,embedding_size],0.0,0.01))
#weights['feature_bias']是FM中的一次项的权重。
weights['feature_bias'] = tf.Variable(tf.random_normal([feature_size,1],0.0,1.0))

embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'],feat_index) # N x field_size x k
feat_value = tf.reshape(feat_value,shape=[-1,field_size,1])

#这里相当于对实值特征进行了一次repeat操作，由1个数repeat为一个k维向量。该实值K维向量乘以embeddings，相当于先进行了一次线性变换，这样做是有好处的，将实数值映射到与embeddings同一向量空间中，利于模型训练。
embeddings = tf.multiply(embeddings,feat_value)
```

## DNN part
```
y_deep = tf.reshape(embeddings, shape=[-1,self.field_size * self.embedding_size]) #Flatten
y_deep = tf.layers.dense(y_deep, activation="relu", use_bias=True)
```

## FM part
```
# 1-order
y_first_order = tf.nn.embedding_lookup(weights['feature_bias'],feat_index)
y_first_order = tf.reduce_sum(tf.multiply(y_first_order,feat_value),2) # None * f

# 2-order
summed_features_emb = tf.reduce_sum(embeddings,1) # None * k
summed_features_emb_square = tf.square(summed_features_emb) # None * k

squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb,1) # None * k

fm_second_order = 0.5 * tf.subtract(summed_features_emb_square,squared_sum_features_emb)
```

## 训练
```
concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=label))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
```

# DCN

在DeepFM中，FM部分只用到了2-order的特征交互，deep部分由于是隐式的构造cross features，所以对于某些类型的特征交互，并不能有效的学习。DCN（Deep & Cross Network），仍然和DeepFM一样沿用wide & deep架构，deep部分仍然是MLP捕获高纬度非线性特征交互，而wide部分改成了Cross Network，其每一层显示应用了特征交叉，事实上，正式由于Cross Network的特殊网络结构，使得the degree of cross features to grow with layer depth。DCN模型架构如下所示：
![DCN](/img/DCN-01.png)
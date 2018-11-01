---
layout:     post   				    # 使用的布局
title:      10.0 softmax回归 				# 标题 
# subtitle:   Hello World, Hello Blog # 副标题
date:       2018-08-13 				# 时间
author:     子颢 						# 作者
# header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 机器学习
    - softmax回归
---

# 算法原理

softmax回归（softmax regression）可以直接支持多分类，而不用当成多个二分类处理。逻辑回归是线性回归套以Sigmoid函数，softmax回归是线性回归套以Softmax函数，即通过线性回归计算出属于每一个类的得分值score，然后用softmax函数归一化为概率。
逻辑回归中y = x * w，w是一个一维向量，softmax回归中W是一个二维矩阵（n*k：n为特征维度，k为类别个数）。
softmax函数表达式：
![softmax](/img/softmax-01.png)

softmax回归的损失函数为交叉熵损失（Cross Entropy），交叉熵通常用来衡量预估类别概率和真实类别概率的匹配程度。
![softmax](/img/softmax-02.png)
当只有两个类时（K=2，二分类问题），交叉熵损失跟逻辑回归的损失函数相等，所以逻辑回归是softmax回归在二分类时的特例，softmax回归是逻辑回归在多分类情况下的推广。

# 模型训练

## sklearn版本

代码地址 <a href="https://github.com/qianshuang/ml-exp" target="_blank">https://github.com/qianshuang/ml-exp</a>
```
def train():
    print("start training...")
    # 处理训练数据
    # train_feature, train_target = process_file(train_dir, word_to_id, cat_to_id)  # 词频特征
    train_feature, train_target = process_tfidf_file(train_dir, word_to_id, cat_to_id)  # TF-IDF特征
    # 模型训练
    model.fit(train_feature, train_target)


def test():
    print("start testing...")
    # 处理测试数据
    test_feature, test_target = process_file(test_dir, word_to_id, cat_to_id)
    # test_predict = model.predict(test_feature)  # 返回预测类别
    test_predict_proba = model.predict_proba(test_feature)    # 返回属于各个类别的概率
    test_predict = np.argmax(test_predict_proba, 1)  # 返回概率最大的类别标签

    # accuracy
    true_false = (test_predict == test_target)
    accuracy = np.count_nonzero(true_false) / float(len(test_target))
    print()
    print("accuracy is %f" % accuracy)

    # precision    recall  f1-score
    print()
    print(metrics.classification_report(test_target, test_predict, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    print(metrics.confusion_matrix(test_target, test_predict))


if not os.path.exists(vocab_dir):
    # 构建词典表
    build_vocab(train_dir, vocab_dir)

categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

# kNN
# model = neighbors.KNeighborsClassifier()
# decision tree
# model = tree.DecisionTreeClassifier()
# random forest
# model = ensemble.RandomForestClassifier(n_estimators=10)  # n_estimators为基决策树的数量，一般越大效果越好直至趋于收敛
# AdaBoost
# model = ensemble.AdaBoostClassifier(learning_rate=1.0)  # learning_rate的作用是收缩基学习器的权重贡献值
# GBDT
# model = ensemble.GradientBoostingClassifier(n_estimators=10)
# xgboost
# model = xgboost.XGBClassifier(n_estimators=10)
# Naive Bayes
model = naive_bayes.MultinomialNB()
# logistic regression
model = linear_model.LogisticRegression()   # ovr
model = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs")  # softmax回归

train()
test()
```
运行结果：
```
read_category...
read_vocab...
start training...
start testing...

accuracy is 0.935000

             precision    recall  f1-score   support

         游戏       1.00      0.97      0.99       104
         时政       0.99      0.79      0.88        94
         体育       0.99      0.97      0.98       116
         财经       0.97      0.99      0.98       115
         家居       0.94      0.74      0.83        89
         科技       0.99      0.99      0.99        94
         时尚       1.00      0.89      0.94        91
         教育       0.82      1.00      0.90       104
         娱乐       0.96      1.00      0.98        89
         房产       0.78      0.96      0.86       104

avg / total       0.94      0.94      0.93      1000

Confusion Matrix...
[[101   0   0   0   0   1   0   2   0   0]
 [  0  74   0   3   0   0   0   6   0  11]
 [  0   1 113   0   0   0   0   1   0   1]
 [  0   0   0 114   0   0   0   0   0   1]
 [  0   0   1   0  66   0   0   7   0  15]
 [  0   0   0   0   1  93   0   0   0   0]
 [  0   0   0   0   1   0  81   5   4   0]
 [  0   0   0   0   0   0   0 104   0   0]
 [  0   0   0   0   0   0   0   0  89   0]
 [  0   0   0   0   2   0   0   2   0 100]]
```

## TensorFlow版本

代码地址 <a href="https://github.com/qianshuang/dl-exp" target="_blank">https://github.com/qianshuang/dl-exp</a>
```
W = tf.Variable(tf.truncated_normal([self.config.vocab_size, self.config.num_classes], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]))

with tf.name_scope("score"):
    y_conv = tf.matmul(self.input_x, W) + b
    self.y_pred_cls = tf.argmax(y_conv, 1)  # 预测类别

with tf.name_scope("optimize"):
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=y_conv))
    # 优化器
    self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

with tf.name_scope("accuracy"):
    # 准确率
    correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
    self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```
运行结果：
```
Configuring LR model...
Loading training data...
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.8, Train Acc:  10.16%, Val Loss:    3.0, Val Acc:  10.00%, Time: 0:00:01 *
Iter:     10, Train Loss:    2.2, Train Acc:  25.00%, Val Loss:    2.2, Val Acc:  21.50%, Time: 0:00:02 *
Iter:     20, Train Loss:    1.7, Train Acc:  46.88%, Val Loss:    1.7, Val Acc:  40.40%, Time: 0:00:04 *
Iter:     30, Train Loss:    1.3, Train Acc:  58.59%, Val Loss:    1.4, Val Acc:  55.50%, Time: 0:00:05 *
Iter:     40, Train Loss:    1.1, Train Acc:  61.72%, Val Loss:    1.1, Val Acc:  65.60%, Time: 0:00:06 *
......
......
......
Epoch: 13
Iter:    860, Train Loss:  0.032, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.60%, Time: 0:01:52 
Iter:    870, Train Loss:  0.045, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.60%, Time: 0:01:54 
Iter:    880, Train Loss:  0.053, Train Acc:  99.22%, Val Loss:   0.17, Val Acc:  95.90%, Time: 0:01:56 *
Iter:    890, Train Loss:   0.06, Train Acc:  99.22%, Val Loss:   0.17, Val Acc:  95.90%, Time: 0:01:57 *
Iter:    900, Train Loss:  0.033, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.80%, Time: 0:01:58 
Iter:    910, Train Loss:  0.044, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.80%, Time: 0:01:59 
Iter:    920, Train Loss:  0.049, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.70%, Time: 0:02:00 
Epoch: 14
Iter:    930, Train Loss:  0.044, Train Acc:  99.22%, Val Loss:   0.17, Val Acc:  95.50%, Time: 0:02:02 
Iter:    940, Train Loss:  0.048, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.30%, Time: 0:02:03 
Iter:    950, Train Loss:   0.06, Train Acc:  99.22%, Val Loss:   0.17, Val Acc:  95.50%, Time: 0:02:04 
Iter:    960, Train Loss:  0.041, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  95.70%, Time: 0:02:05 
Iter:    970, Train Loss:  0.046, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  95.60%, Time: 0:02:06 
Iter:    980, Train Loss:  0.021, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  95.50%, Time: 0:02:07 
Iter:    990, Train Loss:   0.03, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  95.60%, Time: 0:02:09 
No optimization for a long time, auto-stopping...
Loading test data...
Testing...
Test Loss:   0.17, Test Acc:  95.90%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         娱乐       0.97      0.99      0.98        89
         时尚       0.95      0.97      0.96        91
         房产       0.90      0.90      0.90       104
         教育       0.95      0.94      0.95       104
         体育       1.00      0.98      0.99       116
         财经       1.00      0.96      0.98       115
         家居       0.92      0.93      0.93        89
         时政       0.91      0.94      0.92        94
         科技       0.99      1.00      0.99        94
         游戏       0.99      0.98      0.99       104

avg / total       0.96      0.96      0.96      1000

Confusion Matrix...
[[ 88   0   0   0   0   0   0   1   0   0]
 [  2  88   0   0   0   0   1   0   0   0]
 [  0   1  94   0   0   0   3   6   0   0]
 [  1   2   2  98   0   0   1   0   0   0]
 [  0   0   0   1 114   0   0   1   0   0]
 [  0   0   3   0   0 110   1   1   0   0]
 [  0   2   2   1   0   0  83   0   0   1]
 [  0   0   3   2   0   0   1  88   0   0]
 [  0   0   0   0   0   0   0   0  94   0]
 [  0   0   0   1   0   0   0   0   1 102]]
Time usage: 0:00:01
```

# 社群

- QQ交流群
	![562929489](/img/qq_ewm.png)
- 微信交流群
	![562929489](/img/wx_ewm.png)
- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
---
layout:     post   				    # 使用的布局
title:      03.0 随机森林 				# 标题 
# subtitle:   Hello World, Hello Blog # 副标题
date:       2018-08-06 				# 时间
author:     子颢 						# 作者
# header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 机器学习
    - 随机森林
---

# 算法原理

集成学习（ensemble leaning）通过构建并结合多个学习器来完成学习任务，通过将多个学习器结合，常常可以获得比单一学习器显著优越的效果和泛化能力。集成学习中的基学习器可以是同质的，也可以是异质的。根据个体学习器的生成方式，目前的集成学习方法大致可分为三大类：一类是Bagging，个体学习器之间不存在强依赖关系，可以同时并行化训练和生成，最终结果通常通过投票机制产出，随机森林是这一类型的代表；另一类是Boosting，个体学习器之间存在强依赖关系，后一学习器依赖前一学习器的结果，，因此必须以序列化形式串行生成，我们下节会讲到的Adaboost和GBDT是这一类型的代表；其实还有第三类，叫Stacking，即将初级学习器的输出次级学习器的输入特征，深层神经网络甚至可以理解为Stacking集成学习的变种。

随机森林（Random Forest）是以决策树为基学习器构建的Bagging集成学习算法，其实现简单、计算开销小、并且在很多现实任务中表现出抢眼的效果。其主要通过样本扰动和属性扰动使得集成学习的泛化性显著提高（与神经网络中的dropout有异曲同工之妙，有效防止过拟合）。样本扰动是指通过对初始训练集采样构建每一棵决策树；属性扰动是指对基决策树的每个节点，分裂时从该节点的属性集合中随机选择k个属性（k一般去log(d,2)，d为属性数量）。
![随机森林](/img/random_forest-01.png)

# 模型训练

代码地址 <a href="https://github.com/qianshuang/ml-exp" target="_blank">https://github.com/qianshuang/ml-exp</a>

```
def train():
    print("start training...")
    # 处理训练数据
    train_feature, train_target = process_file(train_dir, word_to_id, cat_to_id)
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
model = ensemble.RandomForestClassifier(n_estimators=10)    # n_estimators为基决策树的数量，一般越大效果越好直至趋于收敛

train()
test()
```
运行结果：
```
read_category...
read_vocab...
start training...
start testing...

accuracy is 0.875000

             precision    recall  f1-score   support

         娱乐       0.83      0.91      0.87        89
         房产       0.78      0.83      0.80       104
         教育       0.81      0.81      0.81       104
         家居       0.75      0.71      0.73        89
         游戏       0.93      0.95      0.94       104
         时政       0.78      0.79      0.78        94
         时尚       0.94      0.89      0.92        91
         体育       0.98      0.97      0.97       116
         财经       0.95      0.91      0.93       115
         科技       0.99      0.96      0.97        94

avg / total       0.88      0.88      0.88      1000

Confusion Matrix...
[[ 81   0   1   1   4   0   1   1   0   0]
 [  0  86   1   6   0   9   0   0   1   1]
 [  6   4  84   2   1   4   1   0   2   0]
 [  3  10   4  63   1   4   2   0   2   0]
 [  3   0   1   1  99   0   0   0   0   0]
 [  0   6   9   3   0  74   0   1   1   0]
 [  5   0   0   4   1   0  81   0   0   0]
 [  0   0   2   0   0   2   0 112   0   0]
 [  0   4   1   3   0   2   0   0 105   0]
 [  0   0   1   1   1   0   1   0   0  90]]
```

# 社群

- QQ交流群
	![562929489](/img/qq_ewm.png)
- 微信交流群
	![562929489](/img/wx_ewm.png)
- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
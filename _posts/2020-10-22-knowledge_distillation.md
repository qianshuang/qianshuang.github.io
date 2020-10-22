---
layout:     post   				    # 使用的布局
title:      62.0 知识蒸馏			# 标题 
date:       2020-10-22  			# 时间
author:     钱爽 						# 作者
catalog: true 						# 是否归档
tags:								# 标签
    - 深度学习
    - 迁移学习
---

# 模型原理

现在的深度学习模型越来越大，例如BERT，在线下训练时对时间要求不高的话，还可以接受。但是在线上inference时，如果对延迟要求高的话，像BERT这样的大模型，就很难满足要求。因此，需要找到模型压缩的方法。

知识蒸馏被广泛用于模型压缩和迁移学习当中，目的是把大模型或者多个模型ensemble后的知识提炼给小模型。迁移学习是从一个领域迁移到另一个领域，知识蒸馏是将知识从一个大网络迁移到另一个小网络。也就是说，迁移学习中两个网络所面对的数据集通常是不一样的，而知识蒸馏面对的是同一数据集，其目的是让小网络学到大网络中的输入到输出的映射关系（知识）。原理如下图所示：
![KD](/img/KD-01.png)
可以将模型看成是黑盒子，知识可以看成是输入到输出的映射关系。因此，我们可以先训练好一个teacher网络，然后将teacher网络的输出结果q作为student网络的目标，训练student网络，使得student网络的结果p接近q，因此我们可以将损失函数写成Loss = CE(y,p) + a * CE(q,p)，这里CE是（Cross Entropy），y是真实label的onehot编码，q是teacher网络的输出结果，p是student网络的输出结果。

但是，直接使用teacher网络的softmax的输出结果q，可能不大合适。因为一个网络训练好之后，对于正确的答案会有一个很高的置信度，例如在MNIST数据中，对于某个2的输入，对于2的预测概率会很高，而对于2类似的数字，例如3和7的预测概率为1e-6和1e-7，这样的话，teacher网络学到的数据的相似信息（例如数字2和3、7很类似）很难传达给student网络，因为它们的概率值接近0。因此，文章提出了softmax-T，公式如下所示：
![KD](/img/KD-02.png)
这里qi是student网络学习的对象（soft targets），Zj是神经网络softmax前的输出logit。如果将T取1，这个公式就是softmax；如果T接近于0，则最大的值会接近1，其它值会接近0，近似于onehot编码；如果T越大，则输出结果的分布越平缓，相当于做了平滑处理，起到保留相似信息的作用；如果T等于无穷，就基本是一个均匀分布。最终根据上述损失函数对网络进行训练优化。

综上，知识蒸馏可以将一个网络的知识转移到另一个网络，两个网络可以是同构或者异构。做法是先训练一个teacher网络，然后使用这个teacher网络的输出和数据的真实标签去训练student网络。使用知识蒸馏，可以用来将网络从大网络转化成一个小网络，并保留接近于大网络的效果；也可以将多个网络学到的知识转移到一个网络中，使得单个网络的效果接近emsemble的结果。

# 代码实现

具体模型架构如下：
![KD](/img/KD-03.png)
1. 训练大模型。首先我们先离线训练大模型（这一步在上图中并未体现，上图最左部分是使用已训练好的大模型的结果）。
2. 计算大模型输出。训练完大模型之后，我们将计算soft target，不直接计算output的softmax，这一步进行了一个divided by T蒸馏操作。
3. 训练小模型。小模型的训练包含两部分：soft target loss、hard target loss，通过调节λ的大小来调整两部分损失函数的权重。
5. 小模型预测。

## student model

student model为biLSTM。
```
class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)     
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)
        return linear_out, hidden
```

## teacher model

teacher model为BERT，并对最后四层进行微调。
```
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, 192)
        self.fc2 = nn.Linear(192, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers= False)
        out = self.fc(pooled)
        out = F.relu(out)
        out = self.fc2(out)
        return out
```

## 损失函数

损失函数为student model的输出s_logits和teacher model的输出t_logits的MSE损失，与student model的输出和真实label的交叉熵，的加权和。
```
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) + (1 - a) * loss2(t_logits, s_logits)
    return loss
```
---
layout:     post   				    # 使用的布局
title:      40.0 tensorflow高级特性				# 标题 
date:       2018-10-16 				# 时间
author:     子颢 						# 作者
catalog: true 						# 是否归档
tags:								# 标签
    - TensorFlow
    - keras
---

TensorFlow 1.10及以上版本有很多的高级特性，这些高级特性能给我们的编程带来极大的便利，下面我们将一一进行介绍。

# Colaboratory

Colaboratory是google发布的一个托管的Jupyter notebook环境，可以免费使用，它具有以下特点：
1. 完全云端运行。相当于Google在云端帮你申请了一台免费虚拟机，TensorFlow已经预先安装并针对所使用的硬件进行了优化。
2. 在代码单元中设置库和数据依赖项，使用!pip install或!apt-get创建cell。通过这种方式可以让其他人轻松地重现您的设置。
3. 可以与Github一起使用。如果你在Github上有一个很好的ipynb，只需将您的Github路径添加到colab.research.google.com/github/即可，例如colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb将加载位于你自己的Github上的此ipynb，创建一键式链接；你还可以通过File > Save a copy to Github 轻松地将Colab笔记本的副本保存到Github。
4. 共享和协同编辑。Colab笔记本存储在Google云端硬盘中，可以通过协作方式进行共享，编辑和评论，只需单击笔记本右上角的“共享”按钮。
5. GPU硬件加速。默认情况下，Colab笔记本在云端CPU上运行，可以通过Runtime > Change runtime type，然后选择GPU从而使Colab笔记本在云端GPU上运行。
6. 你也可以参考https://research.google.com/colaboratory/local-runtimes.html说明让Colab笔记本使用你的本地机器硬件，这时Colab有权限直接读写本地文件。
7. 如果要使用云端资源，需要将本地训练数据上传到云端。
	```
	from google.colab import files

	# 文件上传
	uploaded = files.upload()

	for fn in uploaded.keys():
	    print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))

	# 文件下载
	with open('example.txt', 'w') as f:
	    f.write('some content')

	files.download('example.txt')
	```
	
注：后面我们的所有案例都将采用Colaboratory。

# keras

keras是一个基于TensorFlow的高级API接口，相当于在TensorFlow的基础上做了一层封装，其中囊括了对TensorFlow特定功能的一级支持，例如eager execution、tf.data pipeline、Estimators，它独特的模块化和组合式的编程风格使得TensorFlow更加易用、可读性更好、对用户更加友好，并且使TensorFlow的可拓展性更强而不牺牲灵活性和性能。

## Sequential model

在keras中，最常用的模型是Sequential model，通过它将a stack of layers串联（chain）起来。
```
model = keras.Sequential()
# 第一层128个神经元的全连接网络
model.add(keras.layers.Dense(128, activation='relu'))
# 第二层64个神经元的全连接网络
model.add(keras.layers.Dense(64, activation='relu'))
# 输出层：Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

# metrics：Used to monitor training
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# data shape:(None, 128)；labels shape:(None, 10)
# validation_data:监控模型在训练过程中在验证集上的性能。每个epoch结束时，display the loss and metrics in inference mode。
model.fit(data, labels, epochs=100, batch_size=32, validation_data=(val_data, val_labels))

model.evaluate(x, y, batch_size=32)
model.predict(x, batch_size=32)
```

## Functional model

Sequential模型是一个简单的stack of layers，丧失了灵活性，并不能表示任意模型。可以通过下面的方式进行拓展：
```
inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor
# A layer instance is callable on a tensor, and returns a tensor.
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# Instantiate the model given inputs and outputs.
model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
```

## Callbacks

Callback是一个传递给模型的对象，用于在训练期间自定义和扩展其行为。
```
callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks, validation_data=(val_data, val_targets))
```
常用的Callback还有：
tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.

## Save and restore

```
# 存储和加载checkpoint（模型参数）
model.save_weights('./my_model')  # Save weights to a TensorFlow Checkpoint file
model.save_weights('my_model.h5', save_format='h5')  # Save weights to a keras HDF5 file
# Restore the model's state, this requires a model with the same architecture.
model.load_weights('my_model')

# 存储和加载Configuration（模型图）
json_string = model.to_json()  # Serialize a model to JSON format
fresh_model = keras.models.model_from_json(json_string)  # Recreate the model (freshly initialized)

yaml_string = model.to_yaml()  # Serializes a model to YAML format
fresh_model = keras.models.model_from_yaml(yaml_string)  # Recreate the model

# 存储和加载整个模型（参数+图）
model.save('my_model.h5')  # Save entire model to a HDF5 file
model = keras.models.load_model('my_model.h5')  # Recreate the exact same model, including the optimizer.
```
大家不防将我们之前CNN文本分类的案例用keras进行改写，然后对比一下看keras到底有多简洁。

# Eager Execution

TensorFlow的Eager Execution是一种命令式编程环境，可立即evaluate ops，而无需构建图，即ops会立即返回具体的值，而不是构建以后再运行计算图。这样使我们的代码更简洁，将Python代码与TensorFlow无缝结合，便于轻松地使用TensorFlow开发和调试模型。
```
import tensorflow as tf

# 启用Eager Execution会改变TensorFlow ops的行为方式，现在它们会立即评估并将值返回给Python。也就是说tf.Tensor对象会引用具体的值，而不是指向计算图中的node的符号句柄，因此使用print()或debug程序可以很容易的检查结果。
tf.enable_eager_execution()  # 启用eager_execution
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print(m)  # => [[4.]]

# 启用Eager Execution，Numpy与tensor就可以无缝隐式互相转换了，真是妈妈再也不用担心我的学习
a = tf.constant([[1, 2],[3, 4]])
print(a)
# => tf.Tensor([[1 2][3 4]], shape=(2, 2), dtype=int32)

b = tf.add(a, 1)
print(b)  # => tf.Tensor([[2 3][4 5]], shape=(2, 2), dtype=int32)

print(a * b)  # Operator overloading is supported
# => tf.Tensor([[ 2  6][12 20]], shape=(2, 2), dtype=int32)

import numpy as np

c = np.multiply(a, b)
print(c)  # => [[ 2  6][12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())  # => [[1 2][3 4]]
```

# 社群

- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
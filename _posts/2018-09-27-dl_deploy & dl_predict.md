---
layout:     post   				    # 使用的布局
title:      36.0 深度学习模型部署及在线预测				# 标题 
date:       2018-09-27 				# 时间
author:     子颢 						# 作者
catalog: true 						# 是否归档
tags:								#标签
    - 深度学习
    - GPU
    - 模型部署
    - 在线预测
---

# 模型导出

我们在前面的内容中讲到过，可以使用tf.train.Saver.save()和tf.train.Saver.restore()方法保存和恢复模型变量，但是这只是在模型训练过程中用来做checkpoint，保存的也只是模型的变量。只有导出整个模型（除了模型变量、还包括模型计算图和图的元数据），才能做模型部署和在线预测，这时就必须使用SavedModel（也可以导出Session Bundle，但是TensorFlow1.0后被废弃）来保存和加载模型。使用SavedModel做模型导出的代码如下：
```
print('Configuring CNN model...')
config = TCNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(total_dir, vocab_dir)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

# 导出资源文件：词典表
print('start jsondump word_to_id...')
word_to_id_json = json.dumps(word_to_id, ensure_ascii=False)
word_to_id_json_dir = os.path.join(base_dir, 'word_to_id.json')
open_file(word_to_id_json_dir, mode='w').write(word_to_id_json)
print('finish jsondump word_to_id...')

# 导出资源文件：label标签
print('start jsondump cate_to_id...')
id_to_cate = {v: k for k, v in cat_to_id.items()}
cate_to_id_json = json.dumps(id_to_cate, ensure_ascii=False)
cate_to_id_json_dir = os.path.join(base_dir, 'label_index.json')
open_file(cate_to_id_json_dir, mode='w').write(cate_to_id_json)
print('finish jsondump cate_to_id...')

config.vocab_size = len(words)
config.num_classes = len(categories)
model = TextCNN(config)

# 训练模型
train()
# 测试
test()

# 导出模型
export_path = 'export'
print('Exporting trained model to: ', export_path)
session_export = tf.Session()
# session_export.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables(), sharded=True)
saver.restore(sess=session_export, save_path=save_path)  # 读取保存的模型
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input_x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(model.y_pred_cls)
tensor_info_score = tf.saved_model.utils.build_tensor_info(model.y_sore)
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'texts': tensor_info_x},
        outputs={'pred_cates': tensor_info_y,
                 'pred_score': tensor_info_score},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    session_export, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_categories': prediction_signature
    },
    legacy_init_op=legacy_init_op)
builder.save()
print('Done exporting!')
```
导出的模型文件目录结构如下：
```
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb
```
variables就是模型变量，saved_model.pb即为模型计算图定义。

# 模型部署

模型导出完成后，下一步是部署模型为在线服务，这时候需要用到TensorFlow Serving，它能为模型提供对外rpc调用的接口，以实现跨语言调用。但是首先需要安装TensorFlow Serving，安装完成后调用以下命令，即可启动服务：
```
tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path
```
- port：指定RPC服务监听端口号。
- model_name：自定义模型名。
- model_base_path：所导出的模型文件所在目录。
这里推荐使用阿里云的EAS平台做深度学习模型部署，支持弹性部署和在线扩容。

# 在线预测

当使用TensorFlow Serving部署好了模型服务，下一步就可以跨语言的调用服务了。
Python调用：
```
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```
Java调用：
Java调用推荐使用阿里云的EAS平台，它提供了功能强大的调用服务的JavaSDK。

# 社群

- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
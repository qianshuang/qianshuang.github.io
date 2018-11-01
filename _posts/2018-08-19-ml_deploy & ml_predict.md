---
layout:     post   				    # 使用的布局
title:      16.0 机器学习模型部署及在线预测 				# 标题 
date:       2018-08-19 				# 时间
author:     子颢 						# 作者
catalog: true 						# 是否归档
tags:								#标签
    - 机器学习
    - 模型部署
    - 在线预测
    - deploy
    - predict
---

到目前为止，我们训练的传统机器学习模型都只能进行本地预测（本地调用test方法），那么怎么样把我们的模型部署到线上，然后做在线实时预测呢？
1. 我们的模型实际上就是一个y = f(x)函数，x是特征数据，y是预测结果。我们训练模型的目的就是为了得到f(x)函数的参数；
2. 训练完成后需要对参数进行序列化存储，生成模型文件，这一步叫做模型的导出；
3. 模型的部署即加载模型文件并在内存组装f(x)函数提供在线服务；
4. 在线预测即转换线上数据为模型所需要的特征数据格式，然后调用在线服务，生成预测结果。
所有的机器学习包括深度学习框架训练的模型都是按照以上四个步骤进行部署和在线预测的，只是模型文件的不同。

因为scikit-learn已经成为Python最重要的机器学习库（没有之一），并且到目前为止我们所有的机器学习模型都是通过它训练的，下面主要介绍通过sklearn训练的模型的部署方式：
1. 模型训练完成后，直接将模型导出为PMML(Predictive Model Markup Language)文件。注：PMML是数据挖掘的一种通用的规范，它用统一的XML格式来描述我们生成的机器学习模型，无论你的模型是sklearn,R还是Spark MLlib生成的，我们都可以使用相应的方法将其转化为PMML。关于PMML内部格式细节，请参考 <a href="http://dmg.org/pmml/v4-3/GeneralStructure.html" target="_blank">PMML</a>
```
clf = tree.DecisionTreeClassifier()
pipeline = PMMLPipeline([("classifier", clf)])
pipeline.fit(train_feature, train_target)
# 导出PMML模型文件
from sklearn2pmml import sklearn2pmml
sklearn2pmml(pipeline, "DecisionTree.pmml", with_repr = True)
```
或者在Python端先将模型序列化为pickle，再在Java端将pickle文件转为pmml：
```
clf = tree.DecisionTreeClassifier()
pipeline = PMMLPipeline([("classifier", clf)])
pipeline.fit(train_feature, train_target)
# 导出pickle模型文件
from sklearn.externals import joblib
joblib.dump(pipeline, "pipeline.pkl.z", compress = 9)
```
```
java -jar target/jpmml-sklearn-executable-1.5-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
```
2. 模型加载、部署、服务：实际中，一般将服务封装为Java Web应用或RPC服务，在应用内部加载模型，部署服务。注：JPMML是一个强大的包含模型导出、加载、部署等一条蛇服务的工具包。
```
private Evaluator loadPmml(){
    InputStream is = new FileInputStream("D:/demo.pmml");
    PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
    ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
    Evaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
    return evaluator;
}
private int predict(Evaluator evaluator,int a, int b, int c, int d) {
    Map<String, Integer> data = new HashMap<String, Integer>();
    data.put("x1", a);
    data.put("x2", b);
    data.put("x3", c);
    data.put("x4", d);
    List<InputField> inputFields = evaluator.getInputFields();
    //过模型的原始特征，从画像中获取数据，作为模型输入
    Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
    for (InputField inputField : inputFields) {
        FieldName inputFieldName = inputField.getName();
        Object rawValue = data.get(inputFieldName.getValue());
        FieldValue inputFieldValue = inputField.prepare(rawValue);
        arguments.put(inputFieldName, inputFieldValue);
    }

    Map<FieldName, ?> results = evaluator.evaluate(arguments);
    List<TargetField> targetFields = evaluator.getTargetFields();

    TargetField targetField = targetFields.get(0);
    FieldName targetFieldName = targetField.getName();

    Object targetFieldValue = results.get(targetFieldName);
    System.out.println("target: " + targetFieldName.getValue() + " value: " + targetFieldValue);
    int primitiveValue = -1;
    if (targetFieldValue instanceof Computable) {
        Computable computable = (Computable) targetFieldValue;
        primitiveValue = (Integer)computable.getResult();
    }
    System.out.println(a + " " + b + " " + c + " " + d + ":" + primitiveValue);
    return primitiveValue;
}
```

PMML的确是跨平台的利器，但是也会存在一些问题：
1. PMML为了满足跨平台通用性，牺牲了很多平台独有的优化，所以很多时候我们用算法库自己的保存模型的API得到的模型文件，要比生成的PMML模型文件小很多；
2. PMML文件加载速度也比算法库自己独有格式的模型文件加载慢很多。
3. PMML加载得到的模型和算法库自己独有的模型相比，预测会有一点点的偏差，当然这个偏差并不大。比如某一个样本，用sklearn的决策树模型本地预测为类别1，但是如果我们把这个决策树导出为PMML，并用JAVA加载后，预测有较小的概率出现预测的结果不为类别1；
4. 对于超大模型，比如大规模的集成学习模型（xgboost、随机森林等）以及神经网络，生成的PMML文件很容易得到几个G，甚至上T，这时使用PMML文件加载预测就不太合适了，此时推荐为模型建立一个专有的环境，就没有必要去考虑跨平台了。

更多参考：
- <a href="http://www.naodongopen.com/918.html" target="_blank">Java跨语言调用Python sklearn模型</a>
- <a href="https://github.com/jpmml" target="_blank">jpmml</a>
- <a href="https://help.aliyun.com/document_detail/45395.html" target="_blank">阿里云pai平台机器学习模型部署</a>


# 社群

- QQ交流群
	![562929489](/img/qq_ewm.png)
- 微信交流群
	![562929489](/img/wx_ewm.png)
- 微信公众号
	![562929489](/img/wxgzh_ewm.png)
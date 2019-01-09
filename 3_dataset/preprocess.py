#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# 特征提取中数字型和文本型特征的提取最为常见
# 数字型特征提取
# 数字型特征可以直接作为特征，但对于一个多维特征，某一个特征的取值范围特别大，很可能导致其他特征对结果的影响被忽略。预处理

## 数字特征提取
# 标准化 Z-Score,或者去除均值和方差缩放,(X-mean)/std  计算时对每个属性/每列分别进行
# 将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
# 使用sklearn.preprocessing.scale()函数，可以直接将给定数据进行标准化。
from sklearn import preprocessing
import numpy as np
X = np.array([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]])
X_scaled = preprocessing.scale(X)
print('X_scaled', X_scaled)
# 处理后的数据均值和方差
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# 使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
scaler = preprocessing.StandardScaler().fit(X)
print(scaler.mean_)
print(scaler.scale_)
X_scaled = scaler.transform(X)  # 标准化
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# 正则化
# 正则化的过程是将每个样本缩放到单位范数（每个样本的范数为1），如果后面要使用如二次型（点积）或者其它核方法计算两个样本之间的相似性这个方法会很有用。
# Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
# p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
# 该方法主要应用于文本分类和聚类中。例如，对于两个TF-IDF向量的l2-norm进行点积，就可以得到这两个向量的余弦相似性。
X = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')
print('X_normalized', X_normalized)

# 可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换：
normalizer = preprocessing.Normalizer().fit(X)
print(normalizer)
print(normalizer.transform(X))

# 归一化 将属性缩放到一个指定的最大和最小值（通常是1-0）之间，这可以通过preprocessing.MinMaxScaler类实现。
# 1、对于方差非常小的属性可以增强其稳定性。
# 2、维持稀疏矩阵中为0的条目。
X_train = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print('X_train_minmax', X_train_minmax)

##文本特征提取
# 本质上是做单词切分，不同的单词当作一个新的特征
# 键值city具有多个取值，“Dubai”、“London”和“San Fransisco”，直接把每个取值作为新的特征即可。
# 键值temperature是数值型，可以直接作为特征使用。
measurements = [{'city': 'Dubai', 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

# 文本特征提取有2个非常重要的模型
# 词集模型：单词构成的集合，集合中的每个元素只有一个，即词集中的每个单词就只有一个
# 词袋模型：如果一个单词在文档中出现不止一次，并统计其出现的次数（频数）
# 两者本质区别是，词袋在词集的基础上增加了频率的维度。词集只关注有和没有，词袋还关注有几个。
# 假设我们要对一篇文章进行特征化，最常见的方式就是词袋
from sklearn.feature_extraction.text import CountVectorizer
# 实例化分词对象
vectorizer = CountVectorizer(min_df=1)
print(vectorizer)
# 将文本进行词袋处理
corpus = ['This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document , this?']
X = vectorizer.fit_transform(corpus)
print(X)
# 获取对应的特征名称
print(vectorizer.get_feature_names())
# 获取词袋数据
print(X.toarray())
# 定义词袋的特征空间加词汇表vocabulary
vocabulary = vectorizer.vocabulary_
print(vocabulary)
# 针对其他文本进行词袋处理时，可以直接使用现有的词汇表：
# new_vectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary)
# TensorFlow中有类似实现：
from sklearn.feature_extraction.text import CountVectorizer
# MAX_DOCUMENT_LENGTH = 100
# vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
# x_train = np.array(list(vocab_processor.fit_transform(x_train)))
# x_test = np.array(list(vocab_processor.transform(x_test)))

# 数据读取
# TensorFlow提供了非常便捷的方式从CSV文件中读取数据集。
# 加载对应的函数库：
import tensorflow as tf
import numpy as np
# 从CSV文件中读取数据：
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=" iris_training.csv",
    target_dtype=np.int,
    features_dtype=np.float32)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
# 其中各个参数定义为：
# filename，文件名；
# target_dtype，标记数据类型；
# features_dtype，特征数据类型。
# 访问数据集合的特征以及标记的方式为：
x=training_set.data
y=training_set.target


# 效果验证
# 效果验证是机器学习非常重要的一个环节，最常使用的是交叉验证。
# 以SVM为例，导入SVM库以及Scikit-Learn自带的样本库datasets：
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
print(clf.score(X_test, Y_test))

# K折交叉验证，就是初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。
# 交叉验证重复K次，每个子样本验证一次，平均K次的结果或者使用其他结合方式，最终得到一个单一估测。
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
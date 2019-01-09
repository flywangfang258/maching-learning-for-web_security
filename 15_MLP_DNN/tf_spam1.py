#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


import tensorflow as tf
from tensorflow.contrib.learn import infer_real_valued_columns_from_input, DNNClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0.778,0,0,3.756,61,278,1
def load_SpamBase(filename):
    # 加载数据集
    x = []
    y = []
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            v = line.split(',')
            y.append(int(v[-1]))
            t = []
            for i in range(57):
                t.append(float(v[i]))
            t = np.array(t)
            x.append(t)

    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print(x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


def gnb_main():
    x_train, x_test, y_train, y_test = load_SpamBase("../data/spambase/spambase.data")
    gnb = GaussianNB()
    y_predict = gnb.fit(x_train, y_train).predict(x_test)
    score = metrics.accuracy_score(y_test, y_predict)
    print('Accuracy: {0:f}'.format(score))


def dnn_main():
    x_train, x_test, y_train, y_test = load_SpamBase("../data/spambase/spambase.data")
    feature_columns = infer_real_valued_columns_from_input(x_train)
    print(feature_columns)
    # hidden_units = [30, 10],表明具有两层隐藏层，每层节点数分别为30和10
    classifier = DNNClassifier(feature_columns=feature_columns, hidden_units=[30, 10], n_classes=2)
    # steps=500表明训练500个批次，batch_size=10表明每个批次有10个训练数据。
    # 一个epoch指的是使用全部数据集进行一次训练。进行训练时一个epoch可能更新了若干次参数。epoch_num为指定的epoch次数。
    # 一个step或一次iteration指的是更新一次参数，每次更新使用数据集中的batch_size个数据。
    # 注意: 使用相同的数据集，epoch也相同时，参数更新此时不一定是相同的，这时候会取决于batch_size。
    # iteration或step的总数为(数据总数 / batch_size + 1) * epoch_num
    # 每个epoch都会进行shuffle，对要输入的数据进行重新排序，分成不同的batch。

    classifier.fit(x_train, y_train, steps=500, batch_size=10)
    y_predict = list(classifier.predict(x_test, as_iterable=True))
    #y_predict = classifier.predict(x_test)
    #print y_predict
    score = metrics.accuracy_score(y_test, y_predict)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  # gnb_main()
    dnn_main()
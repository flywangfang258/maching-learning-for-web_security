#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from KNN5.knn_webshell import load_adfa_training_files, dirlist, load_one_file
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def demo():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    h =0.2

    logreg = LogisticRegression(C=1e5)
    logreg.fit(x, y)
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    # print(np.arange(x_min, x_max, h))
    # print(np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 生成网格采样点
    # print(xx, yy)
    # ravel()用于将meshgrid返回的的坐标集合矩阵拉伸，用于后续处理,np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()
    # print(np.c_[xx.ravel(), yy.ravel()])
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    # print(Z)

    Z = Z.reshape(xx.shape)
    # print(Z)
    plt.figure(1, figsize=(4, 3))
    # plt.pcolormesh()会根据y_predict的结果自动在cmap里选择颜色
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)  # paired表示两个两个相近色彩输出，比如浅蓝、深蓝；浅红、深红；浅绿，深绿这种
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(np.linspace(xx.min(), xx.max(), 6))
    plt.yticks(np.linspace(yy.min(), yy.max(), 6))

    plt.show()


def load_adfa_java_files(rootdir):
    x = []
    y = []
    allfile = dirlist(rootdir, [])
    for file in allfile:
        if re.match(r'../data/ADFA-LD/Attack_Data_Master/Java_Meterpreter_\d+/UAD-Java-Meterpreter*', file):
            x.append(load_one_file(file))
            y.append(1)
    return x, y


if __name__ == '__main__':
    # demo()
    x1, y1 = load_adfa_training_files('../data/ADFA-LD/Training_Data_Master')
    x2, y2 = load_adfa_java_files("../data/ADFA-LD/Attack_Data_Master")
    # print('x1:', x1[0:2])
    # print('x2:', x2[0:2])
    x = x1 + x2
    # print(x[0:4])
    y = y1 + y2
    # print x
    vectorizer = CountVectorizer(min_df=1)  # 将文本中的词语转换为词频矩阵, a[i][j]，它表示j词在i类文本下的词频
    x = vectorizer.fit_transform(x)  # 计算各个词语出现的次数
    # print(x)
    print(vectorizer.get_feature_names())
    x = x.toarray()  # 词频矩阵的结果
    # print(x)
    clf = LogisticRegression()
    mlp = MLPClassifier(hidden_layer_sizes=(150, 50), max_iter=10, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.1)
    scores = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(scores)
    print(np.mean(scores))
    print(cross_val_score(mlp, x, y,n_jobs=-1, cv=3))
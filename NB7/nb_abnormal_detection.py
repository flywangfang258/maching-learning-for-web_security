#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn import  datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
from nltk import FreqDist
import operator
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def demo():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    gnb = GaussianNB()
    y_pred = gnb.fit(x, y).predict(x)
    print("num of mislabeled points out of a total %d points: %d" % (x.shape[0], (y != y_pred).sum()))


def operate_command(filename):
    '''
    依次读取每行操作指令，每100个组成一个操作序列，保存在列表里面
    :param filename: 文件名
    :return: 每100个操作序列列表, 总的命令列表
    '''
    oper_cmd = []
    fr = open(filename).readlines()
    x = []
    dist = []
    i = 0
    for line in fr:
        line = line.strip()
        x.append(line)
        dist.append(line)
        i = i + 1
        if i == 100:
            oper_cmd.append(x)
            # print(oper_cmd)
            x = []
            i = 0

    return oper_cmd, dist


def get_label(filename, index=0):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            x.append(int(line.split()[index]))
    return x


# 进行全量比较，将全部命令去重后形成一个大的向量空间，每个命令代表一个特征
def cmd_feature_new(user_cmd_list, dist):
    cmd_feature = []
    for cmd in user_cmd_list:
        v = [0]*len(dist)
        for i in range(0, len(dist)):
            if dist[i] in cmd:
                v[i] += 1
        cmd_feature.append(v)
    return cmd_feature


if __name__ == '__main__':
    N = 90
    # demo()
    cmd, dist = operate_command("../data/masquerade-data/User2")
    feature_vec = cmd_feature_new(cmd, list(set(dist)))
    labels = get_label('../data/masquerade-data/label.txt', 2)
    y = [0] * 50 + labels
    x_train = feature_vec[0:N]
    y_train = y[0:N]

    x_test = feature_vec[N:150]
    y_test = y[N:150]

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    score = np.mean(y_test == y_pred)*100
    print(score)
    print(cross_val_score(gnb, x_train, y_train, n_jobs=-1, cv=10))
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
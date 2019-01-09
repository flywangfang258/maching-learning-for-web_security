#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


'''
检测XSS
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib


def demo():
    # 创建40个随机的点
    np.random.seed(0)
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    # np.random.randn(20, 2) 返回一个或一组样本，具有标准正态分布。
    x = np.r_[np.random.randn(20, 2)-[2, 2], np.random.randn(20, 2)+[2, 2]]
    y = [0]*20 + [1]*20

    # print(x, y)

    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    # print(w)  # [ 0.90230696  0.64821811]
    a = -w[0]/w[1]    # a可以理解为斜率
    xx = np.linspace(-5, 5)
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # 在指定的间隔内返回均匀间隔的数字。
    # print(xx)
    yy = a * xx - (clf.intercept_[0])/w[1]  # 二维坐标下的直线方程

    # plot the parallels to the separating hyperplane that pass through the
    print(clf.support_vectors_)  # get support vectors
    # print(clf.n_support_)  # [2, 1] the number of support vectors
    b = clf.support_vectors_[0]
    yy_down = a*xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a*xx+(b[1]-a*b[0])
    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, facecolors='r')
    plt.axis('tight')
    plt.show()


# web日志特征：url长度、url中包含的第三方域名的个数:http域名个数；https域名个数、敏感字符个数：</>/'/"/、敏感关键字个数:alert/script/onerror/onload/eval/src=

def get_len(url):
    return len(url)


def get_url_count(url):
    if re.search('(http://) | (https://)', url, re.IGNORECASE):
        return 1
    else:
        return 0


def get_evil_char(url):
    return len(re.findall('[<>,\'\"/]', url, re.IGNORECASE))


def get_evil_word(url):
    return len(re.findall("(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)",url,re.IGNORECASE))


def get_last_char(url):
    if re.search('/$', url, re.IGNORECASE):
        return 1
    else:
        return 0


def get_feature(url):
    return [get_len(url), get_url_count(url), get_evil_char(url), get_evil_word(url), get_last_char(url)]


def do_metrics(y_test,y_pred):
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test, y_pred))


def etl(filename, data, isxss):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            f1 = get_len(line)
            f2 = get_url_count(line)
            f3 = get_evil_char(line)
            f4 = get_evil_word(line)
            data.append([f1, f2, f3, f4])
            if isxss:
                y.append(1)
            else:
                y.append(0)
    return data, y


if __name__ == '__main__':
    # demo()
    x = []
    y = []
    etl('../data/xss/xss-200000.txt', x, 1)
    etl('../data/xss/good-xss-200000.txt', x, 0)
    # print(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    joblib.dump(clf, 'xss-svm-200000-module.m')

    clf = joblib.load('xss-svm-200000-module.m')
    y_pred = clf.predict(x_test)
    do_metrics(y_test, y_pred)
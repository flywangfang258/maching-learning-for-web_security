#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def load_kdd99(filename):
    x = []
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line=line.split(',')
            x.append(line)
    return x


def get_apache2andNormal(x):
    '''
    筛选与ddos相关的特征，网络连基本特征，基于时间的网络流量统计特征，基于主机的网络流量统计特征
    筛选标记为‘apache2.和normal.且是http协议的数据
    :param x:
    :return:
    '''
    v = []
    w = []
    y = []
    for x1 in x:
        if (x1[41] in ['apache2.', 'normal.']) and (x1[2] == 'http'):
            if x1[41] == 'apache2.':
                y.append(1)
            else:
                y.append(0)
            # 特征化
            x1 = [x1[0]] + x1[4:8]+x1[22:30]+x1[31:40]
            #x1 = x1[4:8]
            v.append(x1)

    for x1 in v:
        x1 = list(map(float, x1))
        w.append(x1)
    return w, y


if __name__ == '__main__':
    v = load_kdd99('../data/kddcup99/corrected')
    x, y = get_apache2andNormal(v)
    clf = GaussianNB()
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=10))
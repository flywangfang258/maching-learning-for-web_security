#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Rootkit是一种特殊的恶意软件，功能是在安装目标上隐藏自身及指定的文件、进程和网络连接等信息，一般都和木马、后门等其他恶意程序结合使用。
基于KDD99数据集，尝试使用knn算法识别基于telnet连接的rootkit行为。
kdd99数据（41维特征）-> 筛选与rootkit相关特征->基于TCP内容特征->向量化->与rootkit相关的特征向量->knn+10折交叉验证->评估效果
'''
__author__ = 'WF'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
def load_kdd99(filename):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            x.append(line)
    return x


def rootkitAndnormal(x):
    w = []
    y = []
    v = []
    for x1 in x:
        # print(x1)
        if (x1[41] in ['rootkit.', 'normal.']) and (x1[2] == 'telnet'):
            if x1[41] == 'rootkit':
                y.append(1)
            else:
                y.append(0)

            x1 = x1[9:21]
            v.append(x1)

    for x1 in v:
        # print(len(v))
        x2 = list(map(float, x1))
        w.append(x2)
    return w, y


if __name__ == '__main__':
    v = load_kdd99("../data/kddcup99/corrected")
    x, y = rootkitAndnormal(v)
    print(len(x), len(y))
    # print(x, y)
    clf = KNeighborsClassifier(n_neighbors=4)
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=10 ))
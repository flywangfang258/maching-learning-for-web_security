#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pydotplus

def load_kdd99(filename):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            x.append(line)
    return x


def pop3_data(x):
    '''
    筛选标记为guess_passwd.和normal.且是pop3协议的数据
    :param x:
    :return:
    '''
    v = []
    y = []
    for x1 in x:
        if (x1[2] == 'pop_3') and (x1[41] in ['guess_passwd.', 'normal.']):
            v.append(x1)
            if x1[41] == 'guess_passwd.':
                y.append(1)
            else:
                y.append(0)
    print(len(v), len(y))
    return v, y


def feature_select(x):
    '''
    挑选与pop3相关的网络特征以及TCP协议内容的特征作为样本特征
    :param x:
    :return:
    '''
    v = []
    w = []
    for x1 in x:
        x1 = [x1[0]]+x1[4:8]+x1[22:30]
        v.append(x1)
    print(len(v))
    for x1 in v:
        x1 = list(map(float, x1))
        w.append(x1)
    print(len(w))
    return w

if __name__ == '__main__':
    x = load_kdd99('../data/kddcup99/corrected')
    v, y = pop3_data(x)
    print(y)
    x = feature_select(v)
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=10))
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('pop3.pdf')





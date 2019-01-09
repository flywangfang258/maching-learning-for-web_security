#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def load_one_file(filename):
    with open(filename) as f:
        line = f.readline()
        line = line.strip('\n')
    return line


def load_adfa_training_files(rootdir):
    '''
    加载ADFA-LD中的正常样本数据
    :param rootdir:
    :return:
    '''
    x = []
    y = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        if os.path.isfile(path):
            x.append(load_one_file(path))
            y.append(0)
    return x, y


def dirlist(path, allfile):
    '''
    定义遍历目录下的文件函数
    :param path:
    :param allfile:
    :return:
    '''
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = path + '/' + filename
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
        # print(allfile)
    return allfile


def load_adfa_webshell_files(rootdir):
    '''
    从攻击数据集中筛选出和webshell相关的数据
    :param rootdir:
    :return:
    '''
    x=[]
    y=[]
    allfile=dirlist(rootdir, [])
    for file in allfile:
        # print(file)
        if re.match(r"../data/ADFA-LD/Attack_Data_Master/Web_Shell_\d+/UAD-W*", file):
            # print('***********************')
            x.append(load_one_file(file))
            y.append(1)
    return x, y


if __name__ == '__main__':
    x1, y1 = load_adfa_training_files('../data/ADFA-LD/Training_Data_Master')
    x2, y2 = load_adfa_webshell_files("../data/ADFA-LD/Attack_Data_Master")
    # print('x1:', x1[0:2])
    # print('x2:', x2[0:2])
    x = x1 + x2
    # print(x[0:4])
    y = y1 + y2
    # print x
    vectorizer = CountVectorizer(min_df=1)  # 将文本中的词语转换为词频矩阵, a[i][j]，它表示j词在i类文本下的词频
    x = vectorizer.fit_transform(x)  # 计算各个词语出现的次数
    print(x)
    print(vectorizer.get_feature_names())
    x = x.toarray()  # 词频矩阵的结果
    print(x)
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(scores)
    print(np.mean(scores))

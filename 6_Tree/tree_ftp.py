#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import os,re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pydotplus


def load_adfa_training_files(rootdir):
    '''
    加载ADFA-LD中的正常样本
    :param rootdir:
    :return:
    '''
    x = []
    y = []
    list = os.listdir(rootdir)
    for i in range(len(list)):
        path = rootdir + '/' + list[i]
        # print(path)
        if os.path.isfile(path):
            x.append(load_one_file(path))
            y.append(0)
    return x, y


def load_one_file(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
    return line


def dirlist(path, allfile):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = path + '/' + filename
        # print(filepath)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile


def load_adfa_hydra_ftp_files(rootdir):
    '''
    从攻击数据集中筛选与FTP暴力破解相关的数据
    :param rootdir:
    :return:
    '''
    x = []
    y = []
    allfile = dirlist(rootdir, [])
    for file in allfile:
        # print(file)
        if re.match(r"../data/ADFA-LD/Attack_Data_Master/Hydra_FTP_\d+/UAD-Hydra-FTP*", file):
            x.append(load_one_file(file))
            y.append(1)
    return x, y


if __name__ == '__main__':
    x1, y1 = load_adfa_training_files('../data/ADFA-LD/Training_Data_Master')
    # print(x1, y1)
    x2, y2 = load_adfa_hydra_ftp_files('../data/ADFA-LD/Attack_Data_Master')
    # print(x2, y2)
    x = x1 + x2
    y = y1 + y2
    # print(x2[0:2])
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(x)
    print(vectorizer.get_feature_names())
    x = x.toarray()

    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=10))
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('ftp.pdf')


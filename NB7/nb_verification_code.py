#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

'''
朴素贝叶斯在二分类问题上使用广泛，但在多分类问题上表现不如其他算法
'''

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


import pickle
import gzip


def load_data():
    with gzip.open('../data/MNIST/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, valid_data, test_data


if __name__ == '__main__':
    training_data, valid_data, test_data = load_data()
    x1, y1 = training_data
    x2, y2 = test_data
    clf = GaussianNB()
    clf.fit(x1, y1)
    print(cross_val_score(clf, x2, y2, scoring="accuracy"))

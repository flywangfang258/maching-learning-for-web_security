#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import gzip


def load_data():
    with gzip.open('../data/MNIST/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, test_data


if __name__ == '__main__':
    training_data, test_data = load_data()
    x1, y1 = training_data
    x2, y2 = test_data
    clf = LogisticRegression()
    clf.fit(x1, y1)
    print(cross_val_score(clf, x2, y2, scoring="accuracy"))


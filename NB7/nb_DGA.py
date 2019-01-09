#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

#处理域名的最小长度
MIN_LEN = 8

def load_alexa(filename):
    '''
    加载alexa的前1000的域名作为白样本，标记为0
    :param filename:
    :return:
    '''
    domain_list = []
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        # print(row)
        domain = row[1]
        if len(domain) >= MIN_LEN:
            # print(domain)
            domain_list.append(domain)
    return domain_list


def load_dga(filename):
    '''
    加载两个家族的域名
    :param filename:
    :return:
    '''
    domain_list=[]
    # xsxqeadsbgvpdke.co.uk,Domain used by Cryptolocker - Flashback DGA for 13 Apr 2017,2017-04-13,
    # http://osint.bambenekconsulting.com/manual/cl.txt
    with open(filename) as f:
        for line in f:
            domain = line.split(",")[0]
            if len(domain) >= MIN_LEN:
                domain_list.append(domain)
    return domain_list


def nb_dga(model):
    x1_domain_list = load_alexa('../data/alexa/top-1000.csv')
    x2_domain_list = load_dga("../data/alexa/dga-cryptolocke-1000.txt")
    x3_domain_list = load_dga("../data/alexa/dga-post-tovar-goz-1000.txt")

    x_domain_list = np.concatenate((x1_domain_list, x2_domain_list, x3_domain_list))

    y1 = [0]*len(x1_domain_list)
    y2 = [1]*len(x2_domain_list)
    y3 = [2]*len(x3_domain_list)

    y = np.concatenate((y1, y2, y3))
    # 以2-gram处理DGA域名
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r"\w", min_df=1)
    x = cv.fit_transform(x_domain_list).toarray()

    clf = model
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=3))


if __name__ == '__main__':
    # load_alexa('../data/alexa/top-1000.csv')
    nb_dga(GaussianNB())
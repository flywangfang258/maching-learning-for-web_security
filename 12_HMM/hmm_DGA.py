#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

'''
DGA通常是基于当前时间、硬编码常量以及字典，根据一定的算法生成。
'''

import re
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt

# 处理域名的最小长度
MIN_LEN = 10

# 状态个数
N = 8
# 最大似然概率阈值
T = -50

#模型文件名
FILE_MODEL = "hmm_dga.m"


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

def domain2ver(domain):
    ver = []
    for i in range(0, len(domain)):
        ver.append([ord(domain[i])])
    return ver


def train_hmm(domain_list):
    X = [[0]]
    X_lens = [1]
    for domain in domain_list:
        ver = domain2ver(domain)
        np_ver = np.array(ver)
        X = np.concatenate([X, np_ver])
        X_lens.append(len(np_ver))

    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, X_lens)
    joblib.dump(remodel, FILE_MODEL)
    print('model have been saved')

    return remodel


def test_dga(remodel, filename):
    x = []
    y = []
    dga_cryptolocke_list = load_dga(filename)
    for domain in dga_cryptolocke_list:
        domain_ver=domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x, y


def test_alexa(remodel, filename):
    x = []
    y = []
    alexa_list = load_alexa(filename)
    for domain in alexa_list:
        domain_ver = domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x, y


if __name__ == '__main__':
    #domain_list=load_alexa("../data/top-1m.csv")
    domain_list = load_alexa("../data/alexa/top-1000.csv")
    # remodel = train_hmm(domain_list)
    remodel = joblib.load(FILE_MODEL)
    x_3, y_3 = test_dga(remodel, "../data/alexa/dga-post-tovar-goz-1000.txt")
    x_2, y_2 = test_dga(remodel, "../data/alexa/dga-cryptolocke-1000.txt")
    # x_1, y_1 = test_alexa(remodel, "../data/alexa/test-top-1000.csv")
    fig, ax = plt.subplots()
    ax.set_xlabel('Domain Length')
    ax.set_ylabel('HMM Score')
    ax.scatter(x_3, y_3, color='b', label="dga_post-tovar-goz")
    ax.scatter(x_2, y_2, color='g', label="dga_cryptolock")
    #ax.scatter(x_1, y_1, color='r', label="alexa")
    ax.legend(loc='upper right')
    plt.show()
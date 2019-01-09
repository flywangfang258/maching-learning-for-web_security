#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

'''
僵尸网络为了躲避域名黑名单，会使用DGA技术动态生成域名，通过DGA的不同特征可以尝试识别不同的家族群，
以常见的cryptolocker 和post-tovar-goz两个僵尸网络家族为例。
1000个cryptolocker域名
1000个post-tovar-goz域名
alexa前1000的域名
'''

import re
from NB7.nb_DGA import load_alexa, load_dga
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import os
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

# 待提高准确率

# 状态个数
N = 8
# 最大似然概率阈值
T = -50

# 模型文件名
FILE_MODEL = "hmm_xss.m"


def domain2ver(domain):
    ver = []
    for i in range(0, len(domain)):
        ver.append([ord(domain[i])])  # 返回对应的 ASCII 数值，或者 Unicode 数值, 返回值是对应的十进制整数。
    return ver


def train_hmm(domain_list):
    X = [[0]]
    X_lens = [1]
    for domain in domain_list:
        ver = domain2ver(domain)
        np_ver = np.array(ver)
        X = np.concatenate([X, np_ver])
        X_lens.append(len(np_ver))
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, X_lens)
    joblib.dump(remodel, FILE_MODEL)

    return remodel


def test_dga(remodel, domain_list):
    x = []
    y = []
    for domain in domain_list:
        domain_ver = domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x, y


def test_alexa(remodel, x1_domain_list):
    x = []
    y = []
    for domain in x1_domain_list:
        domain_ver = domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x, y



def get_aeiou(domain_list):
    '''
    计算域名的长度，元音字母的比例
    :param domain_list:
    :return:
    '''
    x = []
    y = []
    for domain in domain_list:
        x.append(len(domain))
        count = len(re.findall(r'[aeiou]', domain.lower()))
        count = (0.0 + count)/len(domain)
        y.append(count)
    return x, y


def get_uni_char_num(domain_list):
    '''
    去重后字母数字个数与域名长度的比例
    :param domain_list:
    :return:
    '''
    x = []
    y = []
    for domain in domain_list:
        x.append(len(domain))
        count = len(set(domain))
        count = (0.0 + count)/len(domain)
        y.append(count)
    return x, y


def count2string_jarccard_index(a, b):
    '''
    jarccard系数定义为两个集合交集和并集元素个数的比值， 本例是基于2-gram计算的
    :param a:
    :param b:
    :return:
    '''
    x = set(' '+a[0])
    y = set(' '+b[0])
    for i in range(0, len(a)-1):
        x.add(a[i]+a[i+1])
    x.add(a[-1]+' ')

    for i in range(0, len(b)-1):
        y.add(b[i]+b[i+1])
    y.add(b[-1]+' ')
    return (0.0+len(x-y))/len(x|y)


def get_jarccard_index(a_list, b_list):
    '''
    计算两个域名集合平均jarccard系数的方法
    :param a_list:
    :param b_list:
    :return:
    '''
    x = []
    y = []
    for a in a_list:
        j = 0.0
        for b in b_list:
            j += count2string_jarccard_index(a, b)
        x.append(len(a))
        y.append(j/len(b_list))
    return x, y


def plot(x_1, y_1, x_2, y_2, x_3, y_3, vertical):
    '''
    以域名长度为横轴，元音字母比例为纵轴作图
    :param x_1:
    :param y_1:
    :param x_2:
    :param y_2:
    :param x_3:
    :param y_3:
    :return:
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    ax.set_xlabel('Domain Length')
    ax.set_ylabel(vertical)
    ax.scatter(x_1, y_1, c='b', marker='o', s=50, label='normal')
    ax.scatter(x_2, y_2, c='g', marker='v', s=60, label='dga-cryptolocker')
    ax.scatter(x_3, y_3, c='r', marker='*', s=70, label='dga-tovar-goz')
    # ax.legend(loc='upper right')
    ax.legend(loc='best')
    plt.show()


def concact_fea(x1, x2, x3, x4, x5, i):
    x = np.c_[x1, x2]
    x = np.c_[x, x3]
    x = np.c_[x, x4]
    x = np.c_[x, x5]
    y =[i]*len(x)
    return x, y


if __name__ == '__main__':
    x1_domain_list = load_alexa('../data/alexa/top-1000.csv')
    x_1, y_1 = get_aeiou(x1_domain_list)
    x2_domain_list = load_dga('../data/alexa/dga-cryptolocke-1000.txt')
    x_2, y_2 = get_aeiou(x2_domain_list)
    x3_domain_list = load_dga('../data/alexa/dga-post-tovar-goz-1000.txt')
    x_3, y_3 = get_aeiou(x3_domain_list)
    # plot(x_1, y_1, x_2, y_2, x_3, y_3, 'AEIOU Score')
    x1_, y1_ = get_uni_char_num(x1_domain_list)
    x2_, y2_ = get_uni_char_num(x2_domain_list)
    x3_, y3_ = get_uni_char_num(x3_domain_list)
    # plot(x1_, y1_, x2_, y2_, x3_, y3_, 'UNIQ CHAR NUMBER RATIO')
    x11, y11 = get_jarccard_index(x1_domain_list, x1_domain_list)
    x22, y22 = get_jarccard_index(x2_domain_list, x1_domain_list)
    x33, y33 = get_jarccard_index(x3_domain_list, x1_domain_list)
    # plot(x11, y11, x22, y22, x33, y33, 'JARCCARD INDEX')

    if not os.path.exists(FILE_MODEL):
        remodel = train_hmm(x1_domain_list)
    remodel = joblib.load(FILE_MODEL)
    x_33, y_33 = test_dga(remodel, x3_domain_list)
    x_22, y_22 = test_dga(remodel, x2_domain_list)
    x_11, y_11 = test_alexa(remodel, x1_domain_list)
    # plot(x_11, y_11, x_22, y_22, x_33, y_33, 'HMM Score')
    # print(len(x_1), len(x11), len(x_11), len(y_11))

    # print(x_1[0:5], x1_[0:5], x11[0:5], x_11[0:5])
    # 准备数据
    # x:域名长度、元音字母个数、去重后字母个数与域名长度之比、平均jarccard系数、HMM系数
    x1, y1 = concact_fea(x_1, y_1, y1_, y11, y_11, 0)
    # print(x1)
    x2, y2 = concact_fea(x_2, y_2, y2_, y22, y_22, 1)
    x3, y3 = concact_fea(x_3, y_3, y3_, y33, y_33, 2)

    # x = np.r_[x1, x2]
    # y = np.r_[y1, y2]

    x = np.r_[x1, x2, x3]

    y = np.r_[y1, y2, y3]

    # 使用SVM进行分类

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    from sklearn.preprocessing import StandardScaler  # 标准化

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_x)
    X_test_std = scaler.transform(test_x)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train_std, train_y)
    y_pred = clf.predict(X_test_std)
    print(metrics.accuracy_score(test_y, y_pred))
    print(metrics.confusion_matrix(test_y, y_pred))
    print(classification_report(test_y, y_pred))


# 0.963793103448
# [[177  16   0]
#  [  5 183   0]
#  [  0   0 199]]



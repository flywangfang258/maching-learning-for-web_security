#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'
'''
待改进
'''

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import nltk
import re
import urllib
import html
import joblib
#处理参数值的最小长度
MIN_LEN = 5

# 状态个数
N = 10
# 最大似然概率阈值
T = -5

# 字母
# 数字  1
# <>,:"'
# 其他字符 2
SEN = ['<', '>', ',', ':', '\'', '/', ';', '"', '{', '}', '(', ')']

index_wordbag = 1  # 词袋索引
wordbag = {}  # 词袋

tokens_pattern = r'''(?x)
 "[^"]+"
|http://\S+
|</\w+>
|<\w+>
|<\w+
|\w+=
|>
|\w+\([^<]+\) #函数 比如alert(String.fromCharCode(88,83,83))
|\w+
'''


def demo():
    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                         [0.3, 0.5, 0.2, 0.0],
                         [0.0, 0.3, 0.5, 0.2],
                         [0.2, 0.0, 0.2, 0.6]])
    # The means of each component
    means = np.array([[0.0,  0.0],
                      [0.0, 11.0],
                      [9.0, 10.0],
                      [11.0, -1.0]])
    # The covariance of each component
    # np.identity(2)只能创建方形矩阵
    print(np.identity(2))
    print(np.tile(np.identity(2), (4, 1, 1)))
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    # 参数covariance_type，取值为"full"意味所有的μ,Σ都需要指定。取值为“spherical”则Σ的非对角线元素为0，对角线元素相同。
    # 取值为“diag”则Σ的非对角线元素为0，对角线元素可以不同，"tied"指所有的隐藏状态对应的观测状态分布使用相同的协方差矩阵Σ
    # n_components : 隐藏状态数目
    model = hmm.GaussianHMM(n_components=4, covariance_type="full")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = startprob  # 初始向量
    model.transmat_ = transmat  # 转移矩阵
    model.means_ = means  # 均值
    model.covars_ = covars  # 方差
    # Generate samples
    X, Z = model.sample(500)  # 随机生成一个模型的Z和X

    # Plot the sampled data
    plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
             mfc="orange", alpha=0.7)

    # Indicate the component numbers
    for i, m in enumerate(means):
        plt.text(m[0], m[1], 'Component %i' % (i + 1),
                 size=17, horizontalalignment='center',
                 bbox=dict(alpha=.7, facecolor='w'))
    plt.legend(loc='best')
    plt.show()


def ischeck(str1):
    if re.match(r'^(http)', str1):
        return False
    for i, c in enumerate(str1):
        if ord(c) > 127 or ord(c) < 31:
            return False
        if c in SEN:
            return True
        # 排除中文干扰 只处理127以内的字符
    return False


# 分词
def split_str(line):
    words = nltk.regexp_tokenize(line, tokens_pattern)
    # print(words)
    return words


def load_wordbag(filename, max=100):
    X = [[0]]
    X_lens = [1]
    tokens_list = []
    global wordbag
    global index_wordbag

    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            # url解码   %3c => <  %3e => >
            line = urllib.request.unquote(line)
            # 处理html转义字符 &amp; => &
            line = html.unescape(line)
            if len(line) >= MIN_LEN:
                # print("Learning xss query param:(%s)" % line)
                # 数字常量替换成8
                line, number = re.subn(r'\d+', "8", line)
                # ulr日换成http://u
                line, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:=]+', "http://u", line)
                # 干掉注释
                line, number = re.subn(r'\/\*.?\*\/', "", line)
                # print("Learning xss query etl param:(%s) " % line)
                # print(split_str(line))
                tokens_list += split_str(line)
                # print(tokens_list)

    fredist = nltk.FreqDist(tokens_list)  # 单文件词频
    import operator
    dist_dict = dict(fredist.items())
    sorted_fredist = sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True)
    keys = sorted_fredist[:max]
    # print(keys)
    # print(wordbag)
    for localkey in keys:  # 获取统计后的不重复词集
        if localkey[0] in wordbag.keys():  # 判断该词是否已在词袋中,依次编号
            continue
        else:
            wordbag[localkey[0]] = index_wordbag
            index_wordbag += 1

    print("GET wordbag size(%d)" % len(wordbag))


def main(filename):
    X = [[0]]
    X_lens = [1]
    global wordbag
    global index_wordbag

    with open(filename) as f:
        num = 0
        for line in f:
            num += 1
            line = line.strip('\n')
            # url解码
            line = urllib.request.unquote(line)
            # 处理html转义字符
            line = html.unescape(line)
            vers = []

            if len(line) >= MIN_LEN:
                # print("Learning xss query param:(%s)" % line)
                # 数字常量替换成8
                line, number = re.subn(r'\d+', "8", line)
                # ulr日换成http://u
                line, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", line)
                # 干掉注释
                line, number = re.subn(r'\/\*.?\*\/', "", line)
                # print "Learning xss query etl param:(%s) " % line
                words = split_str(line)
                for word in words:
                    if word in wordbag.keys():
                        vers.append([wordbag[word]])
                    else:
                        vers.append([-1])
                    # print(word, vers)
            np_vers = np.array(vers)
            # print("np_vers:", np_vers, "X:", X)
            # print('np_vers:', np_vers)
            X = np.concatenate((X, np_vers))
            # print(X)
            X_lens.append(len(np_vers))
            # print(X_lens)
    # N = int(sum(X_lens) / num)
    remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
    # print(X)
    # 将范化后的向量X及对应的长度矩阵X_lens输入即可，需要X_lens的原因是参数样本的长度可能不一致，所有需要单独输入
    remodel.fit(X, X_lens)
    joblib.dump(remodel, "xss-train.pkl")

    return remodel


def test(remodel, filename):
    '''
    以黑找黑
    :param remodel:
    :param filename:
    :return:
    '''
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            # url解码
            line = urllib.request.unquote(line)
            # 处理html转义字符
            line = html.unescape(line)

            if len(line) >= MIN_LEN:
                # print  "CHK XSS_URL:(%s) " % (line)
                # 数字常量替换成8
                line, number = re.subn(r'\d+', "8", line)
                # ulr日换成http://u
                line, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", line)
                # 干掉注释
                line, number = re.subn(r'\/\*.?\*\/', "", line)
                # print "Learning xss query etl param:(%s) " % line
                words = split_str(line)
                # print "GET Tokens (%s)" % words
                vers = []
                for word in words:
                    # print "ADD %s" % word
                    if word in wordbag.keys():
                        vers.append([wordbag[word]])
                    else:
                        vers.append([-1])
                np_vers = np.array(vers)
                # print(np_vers)
                # print("CHK SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro, v, line))
                pro = remodel.score(np_vers)

                if pro >= T:
                    print("SCORE:(%d) XSS_URL:(%s) " % (pro, line))
                    # print(line)


def test_normal(remodel, filename):
    '''
    以白找黑
    :param remodel:
    :param filename:
    :return:
    '''
    with open(filename) as f:
        for line in f:
            # 切割参数
            result = urllib.parse.urlparse(line)
            # url解码
            query = urllib.request.unquote(result.query)
            params = urllib.parse.parse_qsl(query, True)

            for k, v in params:
                v = v.strip('\n')
                #print  "CHECK v:%s LINE:%s " % (v, line)

                if len(v) >= MIN_LEN:
                    # print  "CHK XSS_URL:(%s) " % (line)
                    # 数字常量替换成8
                    v, number = re.subn(r'\d+', "8", v)
                    # ulr日换成http://u
                    v, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", v)
                    # 干掉注释
                    v, number = re.subn(r'\/\*.?\*\/', "", v)
                    # print "Learning xss query etl param:(%s) " % line
                    words = split_str(v)
                    # print "GET Tokens (%s)" % words
                    vers = []
                    for word in words:
                        # print "ADD %s" % word
                        if word in wordbag.keys():
                            vers.append([wordbag[word]])
                        else:
                            vers.append([-1])

                    np_vers = np.array(vers)
                    # print np_vers
                    # print  "CHK SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro, v, line)
                    pro = remodel.score(np_vers)
                    print("CHK SCORE:(%d) QUREY_PARAM:(%s)" % (pro, v))
                    if pro < T:
                        print("XSS_URL:(%s) " % v)


if __name__ == '__main__':
    load_wordbag('../data/xss/good-xss-2000.txt', 200)
    # print(wordbag)
    remodel = main('../data/xss/good-xss-2000.txt')
    test_normal(remodel, '../data/xss/xss-good-10-20.txt')
    # test(remodel, '../data/xss/xss-2000.txt')

    # print(urllib.request.unquote("http%3A//blog.51cto.com"))  # http://blog.51cto.com 处理url编码
    # print(html.unescape("http%3A//blog.51cto.com &amp;"))  # http%3A//blog.51cto.com & 处理html转义字符

# dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
#
# # 生成词汇表

#
# vocabSet = set()
# for doc in dataset:
#     vocabSet |= set(doc)
# vocabList = list(vocabSet)
#
# # 根据词汇表生成词集模型
#
# SOW = []
# vec = [0]*len(vocabList)
# for doc in dataset:
#     for i, word in enumerate(vocabList):
#         if word in doc:
#             vec[i] = 1
#     SOW.append(vec)


# 需要支持的词法切分原则为：
#
# 单双引号包含的内容 ‘XSS’
#
# http/https链接 http://xi.baidu.com/xss.js
#
# <>标签 <script>
#
# <>标签开头 <BODY
#
# 属性标签 ONLOAD=
#
# <>标签结尾 >
#
# 函数体 “javascript:alert(‘XSS’);”

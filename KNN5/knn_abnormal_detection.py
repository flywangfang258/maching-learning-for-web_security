
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# print(__doc__)

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from nltk import FreqDist
import operator
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# 训练样本数
N = 90

def demo_knn_unsupervised():
    '''
    无监督学习demo
    :return: 无
    '''
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # 邻居数设置为2， 训练
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    # print(nbrs)

    distances, indices = nbrs.kneighbors(X)  # 每个点的最近邻点是点本身，距离是零。
    print(distances)
    print(indices)
    print(nbrs.kneighbors_graph(X).toarray())  # 产生一个显示相邻点之间连接的稀疏图

def demo_knn_supervised():
    # KNN用于监督学习
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    print(neigh.predict_proba([[0.9]]))



def operate_command(filename):
    '''
    依次读取每行操作指令，每100个组成一个操作序列，保存在列表里面
    :param filename: 文件名
    :return: 每100个操作序列列表, 总的命令列表
    '''
    oper_cmd = []
    fr = open(filename).readlines()
    x = []
    dist = []
    i = 0
    for line in fr:
        line = line.strip()
        x.append(line)
        dist.append(line)
        i = i + 1
        if i == 100:
            oper_cmd.append(x)
            # print(oper_cmd)
            x = []
            i = 0

    return oper_cmd, dist


def most_comm(dist, n):
    '''
    统计最频繁使用的前n个命令和最不频繁使用的50个命令
    :param dist: cmd列表
    :param n: 前n个
    :return: 统计最频繁使用的前n个命令和最不频繁使用的50个命令
    '''
    # 统计最频繁使用的前50个命令和最不频繁使用的前50个命令
    fdist = list(FreqDist(dist).keys())  # 接受list类型的参数，返回词典，key是元素，value是元素出现的次数

    # print(fdist, FreqDist(dist).items())
    # print(FreqDist(dist).most_common(50))

    dist_dict = dict(FreqDist(dist).items())
    # print(dist_dict)
    dist_sorted = sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print(dist_sorted)
    dist_max = set(list(map(lambda x: x[0], dist_sorted[0: n])))
    dist_min = set(list(map(lambda x: x[0], dist_sorted[-n:])))
    return dist_max, dist_min

def featureize(cmd_list):
    '''
    特征化：
    去重操作命令个数；
    最频繁使用的前10个命令
    最不频繁使用的前10个命令
    :param cmd_list: 每100个命令列表
    :return: 特征列表
    '''
    fea_vec = []
    for cmd in cmd_list:
        fea_single = []
        cmd_num = len(set(cmd))
        fea_single.append(cmd_num)
        dist_max, dist_min = most_comm(cmd, 10)
        fea_single.append(dist_max)
        fea_single.append(dist_min)

        fea_vec.append(fea_single)
    return fea_vec


def numeralize(fea_vec, dist):
    '''
    将dist_max和dist_min标量化,和统计的最频繁使用的前50个命令计算重合度
    :param fea_vec: 特征数组
    :return: 数值化的特征数组
    '''
    feature_vec = []
    for fea in fea_vec:
        dist_max, dist_min = most_comm(dist, 50)
        # print(fea[1], fea[1] & dist_max)
        f2 = len(fea[1] & dist_max)
        # print(f2)
        f3 = len(fea[2] & dist_min)
        feature_vec.append([fea[0], f2, f3])
    return feature_vec


def get_label(filename, index=0):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            x.append(int(line.split()[index]))
    return x



# 进行全量比较，将全部命令去重后形成一个大的向量空间，每个命令代表一个特征
# def word_set(filename):
#     dist = []
#     with open(filename) as f:
#         for line in f:
#             line = line.strip()
#             dist.append(line)
#     return set(dist)


def cmd_feature_new(user_cmd_list, dist):
    cmd_feature = []
    for cmd in user_cmd_list:
        v = [0]*len(dist)
        for i in range(0, len(dist)):
            if dist[i] in cmd:
                v[i] += 1
        cmd_feature.append(v)
    return cmd_feature

if __name__ == '__main__':
    # demo_knn_unsupervised()
    # demo_knn_supervised()
    cmd, dist = operate_command("../data/masquerade-data/User2")
    # cmd_l = len(cmd)
    # # print(cmd, cmd_l)
    # # print(dist_max, dist_min)
    # fea_vec = featureize(cmd)
    # # print(fea_vec)
    # feature_vec = numeralize(fea_vec, dist)
    # # print(feature_vec)
    feature_vec = cmd_feature_new(cmd, list(set(dist)))
    labels = get_label('../data/masquerade-data/label.txt', 2)
    y = [0]*50 + labels
    x_train = feature_vec[0:N]
    y_train = y[0:N]

    x_test = feature_vec[N:150]
    y_test = y[N:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)

    y_predict = neigh.predict(x_test)
    print(y_test)
    print(y_predict)
    score = np.mean(y_test == y_predict)*100

    print(score)
    print(cross_val_score(neigh, x_train, y_train, n_jobs=-1, cv=10))
    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))
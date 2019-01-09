#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# scikit中的make_blobs方法常被用来生成聚类算法的测试数据，
# 直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# make_blobs(n_samples=100, n_features=2,centers=3, cluster_std=1.0, \
# center_box=(-10.0, 10.0), shuffle=True, random_state=None)
from NB7.nb_DGA import load_dga,load_alexa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

random_state = 170

def demo():
    n_samples = 1500

    x, y = make_blobs(n_samples=n_samples, random_state=random_state)

    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)

    plt.subplot(111)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.title('hello world!')
    plt.show()


def kmeans_gda():
    x1_domain_list = load_alexa('../data/alexa/top-100.csv')
    x2_domain_list = load_dga("../data/alexa/dga-cryptolocke-1000.txt")
    x3_domain_list = load_dga("../data/alexa/dga-post-tovar-goz-1000.txt")
    x2_domain_list = x2_domain_list[0:50]
    x3_domain_list = x3_domain_list[0:50]
    x_domain_list = np.concatenate((x1_domain_list, x2_domain_list, x3_domain_list))

    y1 = [0]*len(x1_domain_list)
    y2 = [1]*len(x2_domain_list[0:50])
    y3 = [2]*len(x3_domain_list[0:50])

    y = np.concatenate((y1, y2, y3))
    # 以2-gram处理DGA域名
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r"\w", min_df=1)
    x = cv.fit_transform(x_domain_list).toarray()

    model = KMeans(n_clusters=2, random_state=random_state)
    y_pred = model.fit_predict(x)

    print(y_pred)

    tsne = TSNE(learning_rate=100)
    pca = PCA(n_components=2)
    # x = tsne.fit_transform(x)
    x = pca.fit_transform(x)
    for i, label in enumerate(x):
        x1, x2 = x[i]
        if y_pred[i] == 1:
            plt.scatter(x1, x2, marker='o', label='normal', c='r')
        else:
            plt.scatter(x1, x2, marker='<', label='dga', c='g')
    # plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # demo()
    kmeans_gda()
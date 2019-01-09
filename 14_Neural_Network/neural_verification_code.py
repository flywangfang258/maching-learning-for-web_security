#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import gzip
import pickle


def load_data():
    with gzip.open('../data/MNIST/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, test_data


def demo():
    x = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x, y)
    print(clf.predict([[2, 2], [-1, -2]]))


def nn_mnist():
    mnist = input_data.read_data_sets("MNIST_data/")
    # print(mnist)
    # 训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels
    # rescale the data, use the traditional train/test split
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    print(y_test[0:3])
    mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.1)
    # hidden_layer_sizes = (50, 50)，表示有两层隐藏层，第一层隐藏层有50个神经元，第二层也有50个神经元。 在此，隐藏层为1
    mlp.fit(x_train, y_train)
    print(mlp.predict(x_test[0:3]))
    print('层数 :', mlp. n_layers_)
    print('输出的个数 :', mlp.n_outputs_)
    print('迭代次数：', mlp.n_iter_)
    print("Training set score: %f" % mlp.score(x_train, y_train))
    print("Test set score: %f" % mlp.score(x_test, y_test))

    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    # coefs_ 是权重矩阵的列表，其中索引 i 处的权重矩阵表示层 i 和层 i+1 之间的权重
    # intercepts_ 是偏差向量的列表，其中索引 i 处的向量表示添加到层 i+1 的偏差值
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    # ravel	为每个子图设置变量	 ax0,ax1,ax2,ax3=axes.ravel()  创建了4个子图，分别取名为ax0,ax1,ax2和ax3
    # print(len(mlp.coefs_), len(mlp.coefs_[0]), len(mlp.coefs_[0][0]), len(mlp.coefs_[1]), len(mlp.coefs_[1][1]), mlp.coefs_, axes.ravel)
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        # print(len(coef), ax)
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()


if __name__ == '__main__':
    # demo()
    nn_mnist()






























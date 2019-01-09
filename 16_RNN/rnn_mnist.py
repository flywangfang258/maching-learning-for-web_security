#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tflearn
from sklearn import metrics
import numpy as np
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('MNIST_data/', one_hot='True')
# x_train, x_test, y_train, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#
# import gzip, pickle
# with gzip.open('../data/MNIST/mnist.pkl.gz') as fp:
#     training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')

import tflearn.datasets.mnist as mnist
x, y, test_x, test_y = mnist.load_data(one_hot='True')
print(x.shape)


def do_dnn(x, y, test_x, test_y):
    # Building deep neural network
    input_layer = tflearn.input_data(shape=[None, 784])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, keep_prob=0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, keep_prob=0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(x, y, n_epoch=20, validation_set=(test_x, test_y), show_metric=True, run_id='dense_model')


def do_rnn(X, Y, testX, testY):
    # 设置输入参数形状为28*28
    X = np.reshape(X, (-1, 28, 28))
    testX = np.reshape(testX, (-1, 28, 28))
    # 设置使用lstm算法
    net = tflearn.input_data(shape=[None, 28, 28])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)
    # 设置全连接网络
    net = tflearn.fully_connected(net, 10, activation='softmax')
    # 设置输出节点，优化算法使用adam， 损失函数使用categorical_crossentropy
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', name="output1")
    # 创建神经网络实体
    model = tflearn.DNN(net, tensorboard_verbose=2)
    # 调用fit函数训练样本
    model.fit(X, Y, n_epoch=1, validation_set=(testX,testY), show_metric=True, snapshot_step=100)


if __name__ == '__main__':
    # do_dnn(x, y, test_x, test_y)
    do_rnn(x, y, test_x, test_y)


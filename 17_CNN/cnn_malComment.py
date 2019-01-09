#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tflearn
import tensorflow as tf
import tflearn.data_utils as du
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import os

MAX_DOCUMENT_LENGTH = 200
EMBEDDING_SIZE = 50

n_words = 0


def demo():
    import tflearn.datasets.mnist as mnist
    x, y, test_x, test_y = mnist.load_data(one_hot='True')
    print(x.shape)

    x = x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])

    # 按功能划分的零中心将每个样本的中心置零，并指定平均值。如果未指定，则对所有样品评估平均值。
    # Returns : A numpy array with same shape as input. Or a tuple (array, mean) if no mean value was specified.
    x, mean = du.featurewise_zero_center(x)
    test_x = du.featurewise_zero_center(test_x, mean)

    net = tflearn.input_data(shape=[None, 28, 28, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)

    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.1)
    # Training
    model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                        max_checkpoints=10, tensorboard_verbose=0)
    model.fit(x, y, n_epoch=100, validation_set=(test_x, test_y),
              show_metric=True, batch_size=256, run_id='resnet_mnist')


def load_one_file(filename):
    x = ""
    with open(filename) as f:
        for line in f:
            x += line
    return x


def load_files(rootdir, label):
    dir_list = os.listdir(rootdir)
    x = []
    y = []
    for i in range(0, len(dir_list)):
        path = os.path.join(rootdir, dir_list[i])
        if os.path.isfile(path):
            #print "Load file %s" % path
            y.append(label)
            x.append(load_one_file(path))
    return x, y


def load_data():
    x = []
    y = []
    x1, y1 = load_files("../data/movie-review-data/review_polarity/txt_sentoken/pos/",0)
    x2, y2 = load_files("../data/movie-review-data/review_polarity/txt_sentoken/neg/", 1)
    x = x1+x2
    y = y1+y2
    return x, y


def do_cnn(trainX, trainY,testX, testY):
    global n_words
    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    testX = pad_sequences(testX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None, MAX_DOCUMENT_LENGTH], name='input')
    network = tflearn.embedding(network, input_dim=n_words+1, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch = 20, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)


if __name__ == '__main__':
    demo()



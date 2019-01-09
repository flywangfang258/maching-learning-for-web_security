#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tensorflow as tf
from tensorflow.contrib.learn.python import learn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

MAX_DOCUMENT_LENGTH = 200
EMBEDDING_SIZE = 50

n_words = 0


def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            x+=line
    return x


def load_files(rootdir,label):
    list = os.listdir(rootdir)
    x=[]
    y=[]
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            print("Load file %s" % path)
            y.append(label)
            x.append(load_one_file(path))
    return x, y


def load_data():
    '''
    加载数据打标
    :return:
    '''
    x = []
    y = []
    x1, y1 = load_files("../data/movie-review-data/review_polarity/txt_sentoken/pos/", 0)
    x2, y2 = load_files("../data/movie-review-data/review_polarity/txt_sentoken/neg/", 1)
    x = x1+x2
    y = y1+y2
    return x, y


def do_rnn(trainX, testX, trainY, testY):
    global n_words
    # Data preprocessing
    # Sequence padding
    print("GET n_words embedding %d" % n_words)
    trainX = pad_sequences(trainX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    testX = pad_sequences(testX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH])
    net = tflearn.embedding(net, input_dim=n_words, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, run_id="wf")


def do_NB(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    y_predict = gnb.fit(x_train, y_train).predict(x_test)
    score = metrics.accuracy_score(y_test, y_predict)
    print('NB Accuracy: {0:f}'.format(score))


def main(unused_argv):
    global n_words

    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    # 词袋模型特征化
    vp = learn.preprocessing.VocabularyProcessor(max_document_length=MAX_DOCUMENT_LENGTH, min_frequency=1)
    # max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。
    # min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中。
    # vocabulary: CategoricalVocabulary 对象。
    # tokenizer_fn：分词函数
    vp.fit(x)
    x_train = np.array(list(vp.transform(x_train)))
    x_test = np.array(list(vp.transform(x_test)))
    n_words = len(

    )
    print('Total words: %d' % n_words)

    do_NB(x_train, x_test, y_train, y_test)
    do_rnn(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    tf.app.run()

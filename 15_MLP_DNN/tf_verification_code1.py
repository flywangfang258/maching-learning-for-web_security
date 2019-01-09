#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def load_data():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    x_train, x_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_data()
    # 使用placeholder（）函数设置占位符，数据类型为float， shape指定
    x = tf.placeholder('float', [None, 784])
    y_ = tf.placeholder('float', [None, 10])

    # 定义整个系统中的变量，包括W，b，初始化均为0。整个系统的操作为y=softmax(Wx+b)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 定义整个系统的操作函数
    y = tf.nn.softmax(tf.matmul(x, W)+b)

    # 定义衰减函数，这里的衰减函数用交叉熵来衡量，通过梯度下降算法以0.01的学习速率最小化交叉熵。
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 初始化全部变量并定义会话，在TensorFlow中每个会话都是独立的，相互不干扰
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 定义每次训练的数据子集的个数
    batch_size = 100

    # 每次都顺序取出100个数据用于训练，便于梯度下降算法快速瘦脸，整个训练次数取决于整个数据集合的长度及每次训练的个数

    for i in range(int(len(x_train)/batch_size)):
        batch_xs = x_train[(i*batch_size):(i+1)*batch_size]
        batch_ys = y_train[(i*batch_size):(i+1)*batch_size]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # dimension=0 按列找
    # dimension=1 按行找
    # tf.argmax()返回最大数值的下标
    # 通常和tf.equal()一起使用，计算模型准确度
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 使用tf.cast（）先将correct_prediction输出的bool值转换成float32，再求平均。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))





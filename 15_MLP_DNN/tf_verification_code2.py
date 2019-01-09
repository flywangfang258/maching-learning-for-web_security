#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from tf_verification_code1 import load_data
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_train, x_test, y_train, y_test = load_data()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

batch_size = 200
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 定义整个系统的操作函数，其中隐藏层有一层，使用relu函数生成：
hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+b2)

# reduce_sum(
#     input_tensor,  表示输入
#     axis=None,    表示在那个维度进行sum操作。
#     keep_dims=False,   表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。
#     name=None,

#     reduction_indices=None    为了跟旧版本的兼容，现在已经不使用了。
# )
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entroy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(int(len(x_train)/batch_size)):
    batch_xs = x_train[(i*batch_size):((i+1)*batch_size)]
    batch_ys = y_train[(i*batch_size):((i+1)*batch_size)]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))

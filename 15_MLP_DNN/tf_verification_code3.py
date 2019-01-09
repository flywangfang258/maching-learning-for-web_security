#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'
'''待提升'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
x_train, x_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels

learning_rate = 0.3
batch_size = 100

# 定义整个系统中的占位符
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

# 定义整个系统中的变量
n_input = 784
n_hidden_1 = 350
n_hidden_2 = 200
n_hidden_3 = 100
n_classes = 10
W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
b1 = tf.Variable(tf.zeros([n_hidden_1]))

W2 = tf.Variable(tf.zeros([n_hidden_1, n_hidden_2]))
b2 = tf.Variable(tf.zeros([n_hidden_2]))

W3 = tf.Variable(tf.zeros([n_hidden_2, n_hidden_3]))
b3 = tf.Variable(tf.zeros([n_hidden_3]))

W4 = tf.Variable(tf.zeros([n_hidden_3, n_classes]))
b4 = tf.Variable(tf.zeros([n_classes]))
# def multilayer_perceptron(x, weights, biases):
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     layer_3 = tf.nn.relu(layer_3)
#
#     hidden_drop = tf.nn.dropout(layer_3, keep_prob)
#     out_layer = tf.nn.softmax(tf.matmul(hidden_drop, weights['out']) + biases['out'])
#     # out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#     # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer
#
#
# weigths = {
#     'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
#     'h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.zeros([n_hidden_2, n_hidden_3])),
#     'out': tf.Variable(tf.zeros([n_hidden_3, n_classes]))
#     # 'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
#
# biases = {
#     'b1': tf.Variable(tf.zeros([n_hidden_1])),
#     'b2': tf.Variable(tf.zeros([n_hidden_2])),
#     'b3': tf.Variable(tf.zeros([n_hidden_3])),
#     'out': tf.Variable(tf.zeros([n_classes]))
# }

# 定义整个系统的操作函数
hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2)+b2)
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3)+b3)
hidden_drop = tf.nn.dropout(hidden3, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop, W4)+b4)



# pred = multilayer_perceptron(x, weigths, biases)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(mnist.train.num_examples)
for i in range(int(mnist.train.num_examples / batch_size)):
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: x_train, y_: y_train, keep_prob: 0.75})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
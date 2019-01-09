#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tflearn
from tflearn.data_utils import *
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    path = '../data/cityName/US_Cities.txt'
    maxlen = 20
    string_utf8 = open(path, 'r').read()
    x, y, char_idx = string_to_semi_redundant_sequences(string_utf8, seq_maxlen=maxlen, redun_step=3)

    # string_utf8是输入的字符串，格式为“皇太极\n祖大寿\n倪哑巴\n胡桂南\n胡老三崔秋山\n黄真\n崔希敏\n黄二毛子\n曹化淳\n黄须人”，注意\n也是一个字符。
    # seq_maxlen是生成的序列的长度，这里取20。
    # redun_step是步长，就是每隔几个字取一次，这里取3。
    g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)  # 设置损失和优化器
    # 实例化基于RNN的序列生成器，并使用对应的字典
    m = tflearn.SequenceGenerator(g, dictionary=char_idx, seq_maxlen=maxlen, clip_gradients=5.0, checkpoint_path='model_us_cities')
    # 使用随机种子，通过RNN模型随机生成城市的名称
    for i in range(40):
        # 建立生成序列的种子,随机的
        seed = random_sequence_from_string(string_utf8, maxlen)
        # 填充数据进行训练
        m.fit(x, y, validation_set=0.1, batch_size=128, n_epoch=1, run_id='us_cities')
        print("-- TESTING...")
        print("-- Test with temperature of 1.2 --")
        # 调用模型进行数据生成
        # temperature  新颖程度, 越小，自动生成的城市的名称越接近样本中的城市名称，越大越新鲜
        # 0 表示就是样本数据
        # generate(seq_length, temperature=0.5, seq_seed=None, display=False)
        print(m.generate(30, temperature=1.2, seq_seed=seed))
        print("-- Test with temperature of 1.0 --")
        print(m.generate(30, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(30, temperature=0.5, seq_seed=seed))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import  cross_val_score

r_token_pattern = r'\b\w+\b\(|\'\w+\''

def load_file(file_path):
    '''
    将php文件转换成一个字符串
    :param file_path:
    :return:
    '''
    t = ''
    with open(file_path, 'rb') as f:
        for line in f:
            line = line.decode("utf8", "ignore")
            line = line.strip()
            t += line
    return t


def load_files(path):
    '''
    遍历样本集合，将全部PHP文件以字符串形式加载
    :param path:
    :return:
    '''
    files_list = []
    for r, d, files in os.walk(path):
        # 返回三元组(root, dirs, files)
        # root：所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs：是一个list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files：同样是list, 内容是该文件夹中所有的文件(不包括子目录)
        for file in files:
            if file.endswith('.php'):
                file_path = path + file
                # print('load: ', file_path)
                t = load_file(file_path)
                # print(t)
                files_list.append(t)
    return files_list


def vocabulary_set(webshell_files_list):
    '''
    针对黑样本集合，以2-gram生成全局的词汇表
    :param webshell_files_list:
    :return:
    '''
    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error='ignore', token_pattern=r'\b\w+\b', min_df=1)
    x1 = webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
    vocabulary = webshell_bigram_vectorizer.vocabulary_
    y1 = [1]*len(x1)
    return x1, y1, vocabulary


def featurize_normal(vocabulary, wp_files_list):
    '''
    使用黑样本生成的词汇表，将白样本特征化
    :param vocabulary:
    :param wp_files_list:
    :return:
    '''
    wp_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error='ignore', token_pattern=r'\b\w+\b', min_df=1, vocabulary=vocabulary)
    x2 = wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    y2 = [0]*len(x2)
    return x2, y2


# ****************************************************************************

def vocabulary_new(webshell_files_list, r_token_pattern):
    '''
    以1-gram生成全局词汇表，其中1-gram基于函数和字符串常量进行切割
    :param webshell_files_list:
    :param r_token_pattern:
    :return:
    '''
    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(1, 1), decode_error='ignore', token_pattern=r_token_pattern, min_df=1)
    x1 = webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
    y1 = [1]*len(x1)
    vocabulary = webshell_bigram_vectorizer.vocabulary_
    return x1, y1, vocabulary


def featurize_normal_new(vocabulary, wp_files_list, r_token_pattern):
    '''
    使用黑样本生成的词汇表，将白样本特征化
    :param vocabulary:
    :param wp_files_list:
    :return:
    '''
    wp_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error='ignore', token_pattern=r_token_pattern, min_df=1, vocabulary=vocabulary)
    x2 = wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    y2 = [0]*len(x2)
    return x2, y2


if __name__ == '__main__':
    wp_files_list = load_files('../data/wordpress/')
    # print(wp_files_list)
    webshell_files_list = load_files('../data/PHP-WEBSHELL/xiaoma/')
    # x1, y1, vocabulary = vocabulary_set(webshell_files_list)
    # x2, y2 = featurize_normal(vocabulary, wp_files_list)
    x1, y1, vocabulary = vocabulary_new(webshell_files_list, r_token_pattern)
    x2, y2 = featurize_normal_new(vocabulary, wp_files_list, r_token_pattern)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    gnb = GaussianNB()
    print(cross_val_score(gnb, x, y, n_jobs=-1, cv=3))



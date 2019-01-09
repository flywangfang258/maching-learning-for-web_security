#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

'''
@description: Data preparation for deep learning methods on CSIC2010 dataset.
对于原始的请求数据，主要提取GET、POST、PUT请求数据来进行检测。请求数据提取后对数据进行字符串分割，
分割依据HTTP请求特点进行，主要涉及URL中解码，及参数项、键值对、特殊符号的分割

'''

import urllib.parse as ps


def string_spliting(str_input, symbol):
    '''

    :param str_input: 输入字符串str_input
    :param symbol: 分隔符symbol
    :return: 一个新的字符串，该字符串在所有分隔符symbol的前后各增加一个空格
    '''
    str_words = str_input.split(symbol)
    return (' ' + symbol + ' ').join(str_words)


def string_words_spliting(str_input):
    '''
    :param str_input: 字符串str_input
    :return: 由多个分割符分割后的字符串
    '''
    str_ret = str_input
    str_ret = string_spliting(str_ret, '?')
    str_ret = string_spliting(str_ret, '&')
    str_ret = string_spliting(str_ret, '=')
    str_ret = string_spliting(str_ret, '(')
    str_ret = string_spliting(str_ret, ')')
    str_ret = string_spliting(str_ret, '{')
    str_ret = string_spliting(str_ret, '}')
    str_ret = string_spliting(str_ret, '<')
    str_ret = string_spliting(str_ret, '>')
    str_ret = string_spliting(str_ret, '/')
    str_ret = string_spliting(str_ret, '\\')
    str_ret = string_spliting(str_ret, '.')
    str_ret = string_spliting(str_ret, '"')
    str_ret = string_spliting(str_ret, '\'')
    str_ret = string_spliting(str_ret, ';')
    str_ret = string_spliting(str_ret, '@')
    str_ret = string_spliting(str_ret, '~')
    return str_ret


def http_request_extraction(fread, fwrite):
    '''
    函数从读文件中依次读取每行数据，对GET、POST、PUT请求数据先调用urllib.parse中的unquote_plus进行base64解码，
    然后调用string_words_spliting函数分割字符
    :param fread:读文件
    :param fwrite:写文件
    :return:fwrite参数是一个文件指针，将处理后含有空格做分割符的请求命令字符串输出。
    '''
    # extract the http request string
    text = fread.readlines()
    lines = len(text)
    i = 0
    while i < lines:
        line = text[i]
        n = len(line)
        if line.startswith('GET'):
            cmdGET = line[4:n-10]
            cmdStr = ps.unquote_plus(cmdGET)+'\n'
            cmdStr = string_words_spliting(cmdStr).lstrip()
            fwrite.write(cmdStr.encode('utf-8'))
#            print(cmdStr)
            i = i+12
        if line.startswith('POST'):
            cmdPOST = line[5:n-10]
            cmdPOST = cmdPOST + '?' + text[i+14][:-1]
            cmdStr = ps.unquote_plus(cmdPOST)+'\n'
            cmdStr = string_words_spliting(cmdStr).lstrip()
            fwrite.write(cmdStr.encode('utf-8'))
#            print(cmdStr)
        if line.startswith('PUT'):
            cmdPUT = line[4:n-10]
            cmdPUT = cmdPUT + '?' + text[i+14][:-1]
            cmdStr = ps.unquote_plus(cmdPUT)+'\n'
            cmdStr = string_words_spliting(cmdStr).lstrip()
            fwrite.write(cmdStr.encode('utf-8'))
#            print(cmdStr)
        i = i+1

fwriteNTest = open('data/normalTrafficTestFeature.txt','wb')
freadNTest  = open('data/normalTrafficTest/normalTrafficTest.txt',
                   encoding='utf-8')
http_request_extraction(freadNTest, fwriteNTest)
fwriteNTest.close()

fwriteNTrain = open('data/normalTrafficTrainingFeature.txt','wb')
freadNTrain  = open('data/normalTrafficTraining/normalTrafficTraining.txt',
                    encoding='utf-8')
http_request_extraction(freadNTrain, fwriteNTrain)
fwriteNTrain.close()

fwriteATest = open('data/anomalousTrafficTestFeature.txt','wb')
freadATest  = open('data/anomalousTrafficTest/anomalousTrafficTest.txt',
                   encoding='utf-8')
http_request_extraction(freadATest, fwriteATest)
fwriteATest.close()

fwriteAll = open('data/TrafficFeatureAll.txt','wb')
freadNTest  = open('data/normalTrafficTest/normalTrafficTest.txt',
                   encoding='utf-8')
freadNTrain  = open('data/normalTrafficTraining/normalTrafficTraining.txt',
                    encoding='utf-8')
freadATest  = open('data/anomalousTrafficTest/anomalousTrafficTest.txt',
                   encoding='utf-8')
http_request_extraction(freadNTest, fwriteAll)
http_request_extraction(freadNTrain, fwriteAll)
http_request_extraction(freadATest, fwriteAll)
fwriteAll.close()

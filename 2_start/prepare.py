#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import nltk

# 为 NLTK 安装一些组件
# 这个语句需要先执行，否则在执行下面语句时会报错，执行一次之后就不再需要了可以注释掉！
# 在弹出的窗口中，为所有软件包选择下载“全部”，然后单击“下载”。 这会给你所有分词器，分块器，其他算法和所有的语料库。
# C:\Users\wf\AppData\Roaming\nltk_data
# nltk.download()

para = "Hello World! Isn't it good to see you? Thanks for buying this book."
from nltk.tokenize import sent_tokenize
print(sent_tokenize(para))  # 句切分

from nltk.tokenize import word_tokenize
sentence = "At eight o'clock on Tuesday morning. auther didn't feel very good!"
tokens = word_tokenize(sentence)  # 词切分
print(tokens)

tagged = nltk.pos_tag(tokens)   # 词性标签
print(tagged[0:6])

# 中文正则分句
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(".*?[。！？]")
print(tokenizer.tokenize("世界真大。我想去看看！你想去吗？不了。"))

# 标识名词实体
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
# 展现语法树
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()
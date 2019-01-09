#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from apriopri import apriori, generateRules
import re

if __name__ == '__main__':
    myDat = []
    with open("../data/xss/xss-2000.txt") as f:
        for line in f:
            # /discuz?q1=0&q3=0&q2=0%3Ciframe%20src=http://xxooxxoo.js%3E
            index = line.find("?")
            if index > 0:
                line = line[index + 1:len(line)]
                # print line
                tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29', line)
                # print "token:"
                # print tokens
                myDat.append(tokens)
        f.close()

    L, suppData = apriori(myDat, 0.12)
    rules = generateRules(L, suppData, minConf=0.99)
    print(rules)
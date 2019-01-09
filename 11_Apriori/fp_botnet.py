#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
在互联网情况下存在大量的僵尸主机的扫描行为，僵尸主机频繁更换ip，很难通过ip确定僵尸主机。
通过使用FP-growth算法，分析防火墙的拦截日志，挖掘出浏览器user-agent字段和被攻击的目标url之间的关系，
初步确定潜在的僵尸主机。

ip表示攻击源ip，ua表示浏览器的user-agent字段，target表示被攻击的目标url
'''

__author__ = 'WF'
import pyfpgrowth


def demo():
    transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
    # support = 2
    # minconf = 0.7
    patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
    print(rules)


def botnet():
    transactions = []

    with open("../data/KnowledgeGraph/sample7.txt") as f:
        for line in f:
            line = line.strip('\n')
            ip, ua, target = line.split(',')
            print("Add (%s %s %s)" % (ip, ua, target))
            transactions.append([ip, ua, target])

    patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.9)

    print(rules)


if __name__ == '__main__':
    # demo()
    botnet()
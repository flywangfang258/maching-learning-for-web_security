#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import networkx as nx
import matplotlib.pyplot as plt
import re

iplist={}
goodiplist={}
# 相似度
N = 0.5
# 黑客团伙IP最少个数
M = 3
# 黑客IP攻击目标最小个数
R = 2


def process_data():
    '''
    数据脱敏，独立于下面
    :return:
    '''
    filename = "../data/etl-ip-domain-train.txt"
    with open(filename) as f:
        for line1 in f:
            line1 = line1.strip('\n')
            ip, domain = line1.split()
            ip=re.sub(r'\d$', '*', ip)
            domain= re.sub(r'\w{3}$', '*', domain)
            domain = re.sub(r'^\w{3}', '*', domain)
            print("%s\t%s" % (ip, domain))


def get_len(d1,d2):
    '''
    jarccard系数,作为衡量两个IP集合相似度的方式
    :param d1:
    :param d2:
    :return:
    '''
    ds1 = set()
    for d in d1.keys():
        ds1.add(d)

    ds2 = set()
    for d in d2.keys():
        ds2.add(d)
    return len(ds1 & ds2)/len(ds1 | ds2)


filename = "../data/etl-ip-domain-train.txt"
G = nx.Graph()  # 定义一个图

# 逐行读取攻击数据，按照攻击源IP建立hash表，hash表的键值为被攻击的域名
with open(filename) as f:
    for line in f:
        (ip, domain) = line.split("\t")
        if not ip == "0.0.0.0":
            if ip not in iplist.keys():
                iplist[ip] = {}

            iplist[ip][domain] = 1

# 定义阈值R，攻击的域名超过R的IP才列入统计范围
for ip in iplist.keys():
    if len(iplist[ip]) >= R:
        goodiplist[ip] = 1

# 满足阈值的IP导入图数据库
for ip1 in iplist.keys():
    for ip2 in iplist.keys():
        if not ip1 == ip2:
            weight = get_len(iplist[ip1], iplist[ip2])
            if (weight >= N) and (ip1 in goodiplist.keys()) and (ip2 in goodiplist.keys()):
                # 点不存在会自动添加
                G.add_edge(ip1, ip2, weight=weight)


n_sub_graphs = nx.number_connected_components(G)   # 图的连通子图数量
# 提取图中所有连通子图，返回一个列表，默认按照结点数量由大到小排序
sub_graphs = nx.connected_component_subgraphs(G)
print(n_sub_graphs)

for i, sub_graph in enumerate(sub_graphs):
    n_nodes = len(sub_graph.nodes())
    if n_nodes >= M:
        print("Subgraph {0} has {1} nodes {2}".format(i, n_nodes, sub_graph.nodes()))
nx.draw(G, bbox_inches='tight')
# plt.show()
plt.savefig('fig.png', bbox_inches='tight')
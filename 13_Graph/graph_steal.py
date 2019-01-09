#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import networkx as nx
import matplotlib.pyplot as plt


def helloWord():
    G = nx.Graph()
    G.add_node("u1")
    G.add_node("u2")
    G.add_edge("u1", "1.1.1.1")
    G.add_edge("u2", "1.1.1.1")
    nx.draw(G, with_labels=True, node_size=600)
    # plt.show()
    plt.savefig('hello.png', bbox_inches='tight')


def show1():
    '''
    检测疑似账号被盗：用户名、登录IP地址、安装客户端的手机号、对应的手机，唯一
    :return:
    '''
    with open("../data/KnowledgeGraph/sample1.txt") as f:
        G = nx.Graph()
        for line in f:
            line = line.strip('\n')
            uid, ip, tel, activesyncid=line.split(',')
            # 以uid为中心，添加对应的ip、tel、activesyncid节点
            G.add_edge(uid, ip)
            G.add_edge(uid, tel)
            G.add_edge(uid, activesyncid)
        # 可视化知识图谱
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()


def show2():
    '''
    检测疑似撞库攻击：用户名、登录IP地址、登录状态、浏览器ua字段{通常结合ip和ua字段可以在一定程度上标识一个用户或者设备}
    :return:
    '''
    with open("../data/KnowledgeGraph/sample2.txt") as f:
        G = nx.Graph()
        for line in f:
            line=line.strip('\n')
            uid,ip,login,ua=line.split(',')
            G.add_edge(uid, ip)
            G.add_edge(uid, login)
            G.add_edge(uid, ua)
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()

def show3():
    '''
    检测疑似刷单：硬件指纹、APP登陆的用户名、APP的名称、用户行为：下单/接单
    :return:
    '''
    G = nx.Graph()
    with open("../data/KnowledgeGraph/sample3.txt") as f:
        for line in f:
            line=line.strip('\n')
            hid,uid,app=line.split(',')
            G.add_edge(hid, uid)
            G.add_edge(hid, app)
    f.close()

    with open("../data/KnowledgeGraph/sample4.txt") as f:
        for line in f:
            line=line.strip('\n')
            hid,uid,action=line.split(',')
            G.add_edge(hid, uid)
            G.add_edge(hid, action)
    f.close()

    nx.draw(G, with_labels=True, node_size=600)
    plt.show()


if __name__ == '__main__':
    print("Knowledge Graph")
    # helloWord()
    # show1()
    # show2()
    show3()
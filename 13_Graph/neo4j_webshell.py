#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import re
from neo4j.v1 import GraphDatabase, basic_auth


def load_data():
    nodes = {}
    index = 1

    driver = GraphDatabase.driver('bolt://localhost:7687', auth=basic_auth('neo4j', 'passwd'))
    session = driver.session()

    file_object = open('../data/r-graph.txt', 'r')
    try:
        # 逐行读取生成节点及节点之间的关系
        for line in file_object:
            matchObj = re.match(r'(\S+) -> (\S+)', line, re.M|re.I)
            # print(matchObj)
            if matchObj:
                path = matchObj.group(1)
                ref = matchObj.group(2)
                # print(path, ref)
                if path in nodes.keys():
                    path_node = nodes[path]
                else:
                    path_node = "Page%d" % index
                    nodes[path] = path_node
                sql = "create (%s:Page {url:\"%s\" , id:\"%d\",in:0,out:0})" %(path_node, path, index)
                index = index+1
                session.run(sql)
                # print(sql)
                # 把入度和出度作为节点的属性，更新节点的入度出度属性
                if ref in nodes.keys():
                    ref_node = nodes[ref]
                else:
                    ref_node = "Page%d" % index
                    nodes[ref]=ref_node
                sql = "create (%s:Page {url:\"%s\",id:\"%d\",in:0,out:0})" %(ref_node, ref, index)
                index = index+1
                session.run(sql)
                # print(sql)
                sql = "create (%s)-[:IN]->(%s)" %(path_node, ref_node)
                session.run(sql)
                # print(sql)
                sql = "match (n:Page {url:\"%s\"}) SET n.out=n.out+1" % path
                session.run(sql)
                # print(sql)
                sql = "match (n:Page {url:\"%s\"}) SET n.in=n.in+1" % ref
                session.run(sql)
                # print(sql)
    finally:
        file_object.close()

    session.close()


if __name__ == '__main__':
    load_data()





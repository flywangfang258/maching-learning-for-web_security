#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.tree import DecisionTreeClassifier

def demo():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    print(clf.predict([[2, 2]]))


def iris_tree():
    from sklearn.datasets import load_iris
    from sklearn import tree
    import pydotplus
    iris = load_iris()
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')


if __name__ == '__main__':
    demo()
    iris_tree()
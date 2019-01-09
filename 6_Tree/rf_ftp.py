#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from tree_ftp import load_adfa_training_files, load_one_file, load_adfa_hydra_ftp_files
from sklearn.feature_extraction.text import CountVectorizer


def demo():
    x, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(scores, scores.mean())

    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(scores, scores.mean())


if __name__ == '__main__':
    # demo()
    x1, y1 = load_adfa_training_files('../data/ADFA-LD/Training_Data_Master')
    x2, y2 = load_adfa_hydra_ftp_files('../data/ADFA-LD/Attack_Data_Master')
    x = x1+x2
    y = y1+y2
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(x)
    x = x.toarray()

    clf1 = ExtraTreesClassifier()
    clf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf2.fit(x, y)
    print(cross_val_score(clf1, x, y, n_jobs=-1, cv=10))
    print(cross_val_score(clf2, x, y, n_jobs=-1, cv=10))
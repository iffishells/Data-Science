# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:57:14 2019

@author: Taimoor
"""

corpus = ['text text mining is interesting',
          'text mining is the same as data mining',
          'text and data mining have few differences']
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(corpus)
print(X.toarray())
print(vec.get_feature_names())
print(len(vec.get_feature_names()))
print(vec.vocabulary_)
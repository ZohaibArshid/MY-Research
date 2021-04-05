# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:02:45 2021

@author: Zobi Tanoli
"""

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import csv



data = pd.read_csv('eng to french.csv', encoding='latin1')
print(len(data))

txt = data['Mix Sentences']
txt= txt.values.tolist()

len(txt)
sent = data['Sentiment']
sent= sent.values.tolist()

Etype=[]



Etype.append(txt)
Etype.append(sent)
len(Etype)


def simple_split(Etype,y, length, split_mark=0.8):
    if split_mark > 0. and split_mark < 1.0:
        n= int(split_mark*length)
    else:
        n= int(split_mark)
    X_train = Etype[:n].copy()
    #print( X_train)
    X_test = Etype[n:].copy()
    #print( X_test)
    y_train = y[:n].copy()
    #print( y_train)
    y_test = y[n:].copy()
    #print( y_test)
    return X_train,X_test,y_train,y_test

vectorizer = CountVectorizer()
X_train,X_test,y_train,y_test= simple_split(Etype[0],Etype[1],len(Etype[0]))
print(len(X_train),len(X_test),len(y_train),len(y_test))

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#feature_names = vectorizer.get_feature_names()
#print(feature_names)

classifier= svm.SVC(kernel='linear', gamma='auto', C=2).fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(classification_report(y_test, y_predict))


















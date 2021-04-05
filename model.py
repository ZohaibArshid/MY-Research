# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:53:01 2021

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
import nlpaug.augmenter.word as naw

data = pd.read_csv('review_sentiments_dataset.csv')
txt= data['Text']
#txt= txt.values.tolist()
sent = data['Sentiment']


Etype=[]

Etype.append(txt)
Etype.append(sent)

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


augmented_data= []

aug = naw.RandomWordAug()
for i in X_train:
    augmented = aug.augment(i)
    #print(len(augmented))
    augmented_data.append(augmented)
#print(augmented_data)


X_train = vectorizer.fit_transform(augmented_data)

X_test = vectorizer.transform(X_test)


feature_names = vectorizer.get_feature_names()
#print(feature_names)

classifier= svm.SVC(kernel='linear', gamma='auto', C=2).fit(X_train, y_train)
y_predict = classifier.predict(X_test)
#print(type(y_predict))
#print(type(y_test))
print('This is with Word Aug (SynonymAug) only using Count vectorizer and SVM Classifier ')
print(classification_report(y_test, y_predict))






'''
aug = naw.RandomWordAug()
for i in X_train:
    augmented = aug.augment(i)
    #print(len(augmented))
    augmented_data.append(augmented)
    
    


'''





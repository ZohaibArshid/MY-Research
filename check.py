# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:03:02 2021

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
from textaugment import Wordnet



data = pd.read_csv('review_sentiments_dataset.csv')
print(len(data))

txt = data['Text']
txt= txt.values.tolist()

len(txt)
sent = data['Sentiment']
sent= sent.values.tolist()

Etype=[]



Etype.append(txt)
Etype.append(sent)
len(Etype)


'''
vector= CountVectorizer()
vector.fit(txt)

print("Print Vocabulary: " + str(vector.vocabulary)+ '\n\n')

vector.get_feature_names()
print("Feature names: " + str(vector.get_feature_names())+ '\n\n')
print(vector.vocabulary_)

counts = vector.transform(txt)

#print(counts)

model = MultinomialNB().fit(counts, sent)
predicts= model.predict(counts)
print(predicts)

matrix= confusion_matrix(sent, predicts )
print(matrix)

print(classification_report(sent, predicts))
'''


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

train= list(X_train)
#print(train)




# use to augment only training set 
tt=[]
t= Wordnet()
for i in train:
    t1= t.augment(i)
    #print(t1)
    tt.append(t1)
len(tt)
print(tt)

Xtrain = vectorizer.fit_transform(tt)
print(Xtrain)
X_test = vectorizer.transform(X_test)

#print(len(y_train))

feature_names = vectorizer.get_feature_names()
#print(feature_names)

classifier= svm.SVC(kernel='linear', gamma='auto', C=2).fit(Xtrain, y_train)
y_predict = classifier.predict(X_test)
#print(type(y_predict))
#print(type(y_test))
print(classification_report(y_test, y_predict))





# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:08:40 2021

@author: Zobi Tanoli
"""


import numpy as np
import warnings
import sklearn.exceptions
import requests
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import csv
import pandas as pd
import matplotlib.pyplot as plt









df= pd.read_csv('comparision.csv')
len(df)

df.info()

#-------------  SentiStrength -----------

manual=df['Manually labeled']
y_true = manual.values.tolist()
#print(y_true)
#type(y_true)

senti= df['SentiStrength']
sentisten= senti.values.tolist()
len(sentisten)
y_predict = np.array(sentisten).ravel()
#print(len(y_predict))
type(y_predict)


report= classification_report(y_true, y_predict)
print("Senti Strength Result:")
print(report)



#---------------- Sanford core nlp ---------

sanf= df['Sanford Core NLP']
sanford = sanf.values.tolist()
len(sanford)
y_predict = np.array(sanford).ravel()
print(len(y_predict))
type(y_predict)


manual=df['Manually labeled']
y_true = manual.values.tolist()
#print(y_true)
type(y_true)

#sklearn.metrics.confusion_matrix(y_true, y_predict)



report= classification_report(y_true, y_predict)
print("Sanford Core NLP Result:")
print(report)





#------------------ NLTK ------------------

nl= df['NLTK']
nltk = nl.values.tolist()
y_nlpredict = np.array(nltk).ravel()


manual=df['Manually labeled']
y_true = manual.values.tolist()
print(y_true)
type(y_true)

pos=[]
neu=[]
neg=[]

for i in range(len(y_true)):
    if y_true[i] == 'positive':
        pos.append(y_true[i])
    elif y_true[i] == 'negative':
        neg.append(y_true[i])
    elif y_true[i] == 'neutral':
        neu.append(y_true[i])


lst=[len(pos), len(neg), len(neu)]
list1=["Positive", "Negitive", "Neutral"]
num=np.arange(len(list1))
plt.bar(num, lst, align='center', alpha=0.5)
plt.xticks(num, list1)
plt.ylabel('Total number')
plt.show()



report= classification_report(y_true, y_nlpredict)

print("NLTK Result:")
print(report)





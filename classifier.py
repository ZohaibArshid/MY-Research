# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 05:55:21 2020

@author: Zohaib Arshad Tanoli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report




#print(np.version.version)
myjsonfile=  open('tripadvisor.json','r')
myjsonfile1= open('bolt.json','r')
myjsonfile2= open('pipedrive.json','r')
myjsonfile3= open('toggl.json','r')
myjsonfile4= open('transferwise.json','r')
#myjsonfile= open('data/tripadvisor.json','r')
jsondata = myjsonfile.read()
jsondata1 = myjsonfile1.read()
jsondata2 = myjsonfile2.read()
jsondata3 = myjsonfile3.read()
jsondata4 = myjsonfile4.read()
# Parse
myList=[jsondata,jsondata1,jsondata2,jsondata3,jsondata4]
#print(myList)
a=[]
b=[]
pos=[]
neg=[]
neu=[]
Etype=[]
ptype=[]
rew=[]
counter=1;
for x in myList:
    obj= json.loads(x)
    print("File : ", counter)
    print ("number of record in dataset" + str(len(obj)))
    length_of_data= (len(obj[0]['reviews']))
    print (length_of_data)
#print (obj)
#print(obj[0]['reviews'][0]['sentences'])

#def printme(obj):
 # print(obj)
 
#TypeE=[]
#P_typr[]=""
#positive_sent[]=""
#negitive_sent[]=""
#neutral_sent[]=""
    
        
    for i in range(length_of_data):
        if obj[0]['reviews'][i]['sentences'][0]['type'] == 'E' and obj[0]['reviews'][i]['sentences'][0]['sentiment'] != None:
            #TypeE= obj[0]['reviews'][i]['sentences']
            TypeEtxt= obj[0]['reviews'][i]['sentences'][0]['text'] 
            TypeEsent= obj[0]['reviews'][i]['sentences'][0]['sentiment']
            print (TypeEtxt)
            print (TypeEsent)
            a.append(TypeEtxt)
            b.append(TypeEsent)
        #print(TypeE[7])
    '''  
    for i in range(length_of_data):
        if obj[0]['reviews'][i]['sentences'][0]['type'] == 'P':
            TypeP= obj[0]['reviews'][i]['sentences'][0]['sentiment']
            print (TypeP)
            ptype.append(TypeP)
    '''
        
    print("\n Positive Sentiments \n")
    for i in range(length_of_data):
        if obj[0]['reviews'][i]['sentences'][0]['sentiment'] == 'positive':
            Positive = obj[0]['reviews'][i]['sentences'][0]['sentiment']
            #print(Positive)
            pos.append(Positive)
                
    print("\n Negitive Sentiments \n")
    for i in range(length_of_data):
        if obj[0]['reviews'][i]['sentences'][0]['sentiment'] == 'negative':
            Negitive= obj[0]['reviews'][i]['sentences'][0]['sentiment']
            #print (Negitive)
            neg.append(Negitive)
        
    print("\n Neutral Sentiments \n")
    for i in range(length_of_data):
        if obj[0]['reviews'][i]['sentences'][0]['sentiment'] == 'neutral':
            Neutral= obj[0]['reviews'][i]['sentences'][0]['sentiment']
            #print (Neutral)
            neu.append(Neutral)
            
        
        for i in range(length_of_data):
         if obj[0]['reviews'][i]['sentences'][0]['type'] == 'E':
            reviews = obj[0]['reviews'][i]['sentences']
            #print(reviews)
            rew.append(reviews)
    counter=counter+1
    
    # else :
    #    print("type P Reviews") 
          
    #for i in range(241):
    # if obj[0]['reviews'][i]['sentences'][0]['type'] == 'P':
     #   print("type P reviews: " + str(obj[0]['reviews'][i]['sentences'][0]['sentiment']))  
    
    # for i in range(5):
    #  print(obj[0]['reviews'][i]['sentences'][0]['type'])
    #  print(obj[0]['reviews'][i]['sentences'][0]['sentiment'])
    
    # else:
    #   print(obj[0]['reviews'][i]['sentences'][0]['sentiment'])
"""
    
    -------------------------------------
    #Etype= obj[obj['reviews']['type']=='E']
    #Ptype= obj[obj['reviews']['type']=='p']
"""


print("-----------------------------------")
print("Number of positive sentiment")
print(len(pos))

print("-----------------------------------")
print("Number of negitive sentiment")
print(len(neg))

print("-----------------------------------")
print("Number of neutral sentiment")
print(len(neu))

#print("-----------------------------------")
#print("Number of ptype sentiment")
#print(len(ptype))

print("-----------------------------------")
print("Number of Etype sentiment")
print(len(Etype))
print(Etype)

print("-----------------------------------")
print("Number of Etype Sentences")
print(len(rew))






list=[len(pos), len(neg), len(neu)]
list1=["Positive", "Negitive", "Neutral"]
num=np.arange(len(list1))
plt.bar(num, list, align='center', alpha=0.5)
plt.xticks(num, list1)
plt.ylabel('Total number')
plt.show()

'''
vect = CountVectorizer()
vect.fit(Etype)

print("Vocabulary size: {}". format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))

bag_of_words = vect.transform(Etype)
print(bag_of_words)

print("bags_of_words as an array:\n{}".format(bag_of_words.toarray()))
vect.get_feature_names()
'''
#z=[i for i in b if i == None]

#print(z.index(N one))

#dataset=[i for i in b if i is not None]
#print(dataset)
#df = pd.DataFrame([a,b], columns=('Reviews', 'Sentiment', 'App Name'))
Etype.append(a)
Etype.append(b)
#length_of_etype= len(Etype)
#print(len(Etype))
print(str(Etype))
#print(len(b))


def simple_split(Etype,y, length, split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n= int(split_mark*length)
    else:
        n= int(split_mark)
    X_train = Etype[:n].copy()
    print( X_train)
    X_test = Etype[n:].copy()
    print( X_test)
    y_train = y[:n].copy()
    #print( y_train)
    y_test = y[n:].copy()
    #print( y_test)
    return X_train,X_test,y_train,y_test

vectorizer = CountVectorizer()
X_train,X_test,y_train,y_test= simple_split(Etype[0],Etype[1],len(Etype[0]))
print(len(X_train),len(X_test),len(y_train),len(y_test))

#print("Samples per class: {}".format(np.bincount(y_train)))
#print("Samples per class: {}".format(np.bincount(y_test)))
X_train = vectorizer.fit_transform(X_train)
#print(X_train)
X_test = vectorizer.transform(X_test)
#print(X_test)

feature_names = vectorizer.get_feature_names()
print(feature_names)
'''
lg= LogisticRegression()
print(lg)
scores = cross_val_score(lg, X_train, y_train, cv=5)
print("mean cross-validation accuracy: [:.2f]".format(np.mean(scores)))

#X=rew[:len(rew)//2]
#print(len(X))
#y= rew[len(rew)//2:]
#print(len(y))
X=rew[:29975]
y=rew[29975:59950]

'''

#SVC----> support vector classifier
classifier= svm.SVC(kernel='linear', gamma='auto', C=2).fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(type(y_predict))
print(type(y_test))
print(classification_report(y_test, y_predict))









#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
   
path = "C:\\Preema\\Softwares\\Udacity\\ML\\ud120-projects-master\\tools\\"
    
import sys
from time import time
sys.path.append("C:\\Preema\\Softwares\\Udacity\\ML\\ud120-projects-master\\tools\\")
from email_preprocess import preprocess
exec(open(path+'email_preprocess.py').read())



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)


len(features_train[1])

#########################################################

exec(open(path+'email_preprocess_v1.py').read())

features_train, features_test, labels_train, labels_test = preprocess()

len(features_train[1])

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)


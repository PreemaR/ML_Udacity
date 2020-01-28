#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
clf = svm.SVC(kernel='linear')

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)

#########################################################
#Taking 1% of data
from sklearn import svm
clf = svm.SVC(kernel='linear')

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")


from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)

#########################################################
#Taking 1% of data & rbf kernal
from sklearn import svm
clf = svm.SVC(kernel='rbf')

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)


#########################################################
#Taking 1% of data & rbf kernal
from sklearn import svm
clf = svm.SVC(kernel='rbf', C = 10000)

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)

#########################################################
### Full dataset with rbf and optimized C ###
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)
Accuracy


#########################################################
### Full dataset with rbf and optimized C ###
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000)

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("pred time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)
Accuracy

pred[10]
pred[26]
pred[50]


features_train, features_test, labels_train, labels_test = preprocess()
clf = svm.SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
print("Pred time:", round(time()-t0, 3), "s")
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(pred,labels_test)
Accuracy

from collections import Counter
Counter(pred)

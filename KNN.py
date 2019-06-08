# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:38:32 2018

@author: Pranav
"""
#Import modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

#Load the iris dataset
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target

#Optional, to see the data and labels
#print(iris_data)
#print(iris_labels)

#Split the data into training and testing data in 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(iris_data,iris_labels,test_size=0.30)

#Create a KNearestClassifier with value of K=5
classifier=KNeighborsClassifier(n_neighbors=5)

#Fit the data and build the model
classifier.fit(x_train,y_train)

#Predict using the test data
y_pred=classifier.predict(x_test)

#Print the Confusion Matrix
print('Confusion matrix is as follows')
print(confusion_matrix(y_test,y_pred))

#Print Precision and Recall
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))

def new_predict(arr):
    a1 = np.array(arr).reshape(1, -1)
    pred = classifier.predict(a1)
    print(type(pred))
    print(pred[0])

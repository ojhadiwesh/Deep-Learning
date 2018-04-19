# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:59:46 2018

@author: ojhadiwesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('D:\\Ozone(8hr) data.csv')
dataset_test=pd.read_csv('D:\\Deep Learning\\Ozone_target(8hr) data.csv')

X = dataset_train.iloc[:, 1:71].values
y = dataset_test.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier= Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(36,init='uniform', activation='relu', input_dim=70))
classifier.add(Dense(36,init='uniform', activation='relu'))
classifier.add(Dense(1,init='uniform', activation='sigmoid'))

#compile the ann
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


y_pred = classifier.predict(X_test)
y_pred= (y_pred>0.5)

new_prediction= classifier.predict(sc.transform(np.array([[0.0,0,600, 1, 40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#performing a more detailed anlaysis by using KerasClassifier object of keras. 
#identifying the best parameter of the classification
#identifying the best score of the model
#this is a very compute intensive model so fitting the model takes a lot of time
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
classifier= Sequential()
def build_classifier(optimizer):
    classifier.add(Dense(36,init='uniform', activation='relu', input_dim=70))
    classifier.add(Dense(36,init='uniform', activation='relu'))
    classifier.add(Dense(1,init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
f_classifier= KerasClassifier(build_fn=build_classifier)
parameters=     { 'batch_size': [25,32],
                 'nb_epoch':[100,500],
                 'optimizer':['adam', 'rmsprop']}
grid_search= GridSearchCV(estimator=f_classifier, 
                          param_grid= parameters,
                         scoring= 'accuracy',
                          cv=10)
grid_search= grid_search.fit(X_train, y_train)
best_parameters= grid_search.best_params_
best_accuracy= grid_search.best_score_


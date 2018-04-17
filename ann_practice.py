# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:56:28 2018

@author: Diwesh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier= Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))

#compile the ann
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred= (y_pred>0.5)

new_prediction= classifier.predict(sc.transform(np.array([[0.0,0,600, 1, 40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier= KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies= cross_val_score(estimator=classifier, X= X_train, y= y_train, cv=10, n_jobs=-1)

mean=accuracies.mean()
variance= accuracies.std()

# improve the ANN by feature tuning 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.grid_search import GridSearchCV
def build_classifier():
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier= KerasClassifier(build_fn=build_classifier)
parameters=     { 'batch_size': [25,32],
                 'nb_epoch':[100,500],
                 'optimizer':['adam', 'rmsprop']}
grid_search= GridSearchCV(estimator=classifier, 
                          param_grid= parameters,
                         scoring= 'accuracy',
                          cv=10)
grid_search= grid_search.fit(X_train, y_train)
best_parameters= grid_search.best_params_
best_accuracy= grid_search.best_score_

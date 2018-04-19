# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:54:24 2018

@author: Rhuturaj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the training set
dataset_train= pd.read_csv('D:\\Deep Learning\\MSFT_Train.csv')
training_set= dataset_train.iloc[: ,1:2].values
#import the test set data
dataset_test= pd.read_csv('D:\\Deep Learning\\MSFT_Test.csv')
real_stock_price= dataset_test.iloc[: ,1:2].values
#preprocessing
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled= sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and one output
X_train=[]
y_train=[]
for i in range(120, 1247):
    X_train.append(training_set_scaled[i-120:i,0])
    y_train.append(training_set_scaled[i,0])
    
X_train,y_train=np.array(X_train), np.array(y_train)

#reshape
X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#start an RNN
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential


#intialisse the regresson
regressor= Sequential()

#the first layer of the LSTM and droput
regressor.add(LSTM(units=100, return_sequences= True, input_shape=( X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#the second LSTM layer
regressor.add(LSTM(units=100, return_sequences= True))
regressor.add(Dropout(0.2))

#third LSTM layer
regressor.add(LSTM(units=100, return_sequences= True))
regressor.add(Dropout(0.2))

#fourth LSTM layer
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

#fifth LSTM Layer
#fourth LSTM layer
regressor.add(LSTM(units=100))
regressor.add(Dropout(0.2))

#the output layer
regressor.add(Dense(units=1))

#compiling the regressor
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fit the model on the train data
regressor.fit(X_train, y_train, batch_size=30, epochs=100)

#making predictions 
dataset_total= pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs= dataset_total[len(dataset_total)-len(dataset_test)-120:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(120, len(dataset_test)+120):
    X_test.append(inputs[i-120:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price= regressor.predict(X_test)

predicted_stock_price=sc.inverse_transform(predicted_stock_price)


#visualize the prediction against the real price
plt.plot(real_stock_price, color='red', label='Real Microsoft Stock Price (April 2018)')

plt.plot(predicted_stock_price, color='blue', label='Predicted Microsoft Stock Price (April 2018)')

plt.title('Microsoft stock Price Prediction')
plt.xlabel('TIME')
plt.ylabel('Price in $')
plt.legend()
plt.show()







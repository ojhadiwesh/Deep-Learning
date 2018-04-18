# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:42:40 2018

@author: Diwesh
"""
#importing the libraries and packages 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D

#initialize the CNN
classifier= Sequential()

#start the convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

# pooling
classifier.add(MaxPooling2D(pool_size=(2,2), ))

#Flatten 
classifier.add(Flatten())

#Full connection
classifier.add(Dense(128, activation='relu'))
#the output layer
classifier.add(Dense(1, activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

#fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'D:\Study Material\Deep_Learning_A_Z\Convolutional_Neural_Networks\dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'D:\Study Material\Deep_Learning_A_Z\Convolutional_Neural_Networks\dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8000/32,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000/32)
                        
#making a new predition
import numpy as np
import keras.preprocessing.image
test_image= image.load_img('D:\Study Material\Deep_Learning_A_Z\Convolutional_Neural_Networks\dataset\single_prediction\cat_or_god_1.jpg', target_size=(64,64))
test_image= image.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
result= classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'



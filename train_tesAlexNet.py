# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:24:40 2019

@author: anas
"""

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
import tools.image_gen_extended as T
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from CNNArchitecture import getAlexNet
import numpy as np
from keras.optimizers import SGD

from keras import backend as K
import os
from DatasetHandler import load_images_formList
os.environ['CUDA_VISIBLE_DEVICES'] = ''
img_rows = 224
img_cols = 224
num_classes=3
epochs = 5

batch_size = 64
num_of_train_samples = 1463
num_of_test_samples = 257
num_of_valid_samples = 257
train_data_path = 'food_photos'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical'
                                                    )

X_train, X_test, y_train, y_test = train_test_split(train_generator._filepaths,train_generator.classes, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.05, random_state=1)

y_train_cat = to_categorical(y_train,num_classes)
y_test_cat = to_categorical(y_test, num_classes)
X_train=load_images_formList(X_train,img_rows,img_cols)
X_test=load_images_formList(X_test,img_rows,img_cols)
X_val=load_images_formList(X_val,img_rows,img_cols)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen =ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=64, seed=11,shuffle=True)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

#Instantiate an empty model

model =getAlexNet(img_rows,img_cols)
model.add(Dense(num_classes, activation='softmax'))
"""
model = load_model ('model1.h5')
model.load_weights('model_weighte1.h5')
"""
model.summary()
# Compile the model
opt = SGD(lr=.01, momentum=.9)
csv_logger = CSVLogger('model4.log')

def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004
lr_scheduler = LearningRateScheduler(schedule)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X_train, y_train_cat,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test_cat),
          callbacks=[lr_scheduler, csv_logger])

model.save('model1.h5')
model.save_weights('model_weighte1.h5')
""""
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(X_val, len(X_val) // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_val, y_pred))
print('Classification Report')
target_names = ['FrenchFries', 'Pizza', 'VegBurger']
print(classification_report(y_val, y_pred, target_names=target_names))
"""



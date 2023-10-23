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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from CNNArchitecture import getAlexNet
import numpy as np
import numpy
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

#y_train_cat = to_categorical(train_generator.classes,num_classes)
y_train_cat = train_generator.classes

X_train=load_images_formList(train_generator._filepaths,img_rows,img_cols)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=64, seed=11)
X_train = X_train.astype('float32')
X_train /= 255


#Instantiate an empty model


#model.summary()
# Compile the model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset

# split into input (X) and output (Y) variables
X = X_train
Y = y_train_cat
opt = SGD(lr=.01, momentum=.9)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    model = getAlexNet(img_rows, img_cols)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    # Fit the model
    y_train=to_categorical(Y[train],num_classes)
    model.fit(X[train], y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    # evaluate the model
    y_test = to_categorical(Y[test], num_classes)
    scores = model.evaluate(X[test], y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    model.save('model2.h5')
    model.save_weights('model_weighte2.h5')







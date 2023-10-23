import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from CNNArchitecture import getModel1
from TrainingMethod import sampleTraining
#Start
train_data_path = 'treaning'
test_data_path = 'test'
img_rows = 224
img_cols = 224

epochs = 5
batch_size = 16
num_of_train_samples = 480+321+550
num_of_test_samples = 12+80+147
num_classes=3
#Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',shuffle=True)

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',shuffle=True)
# Build model
model= getModel1(img_rows,img_cols)
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Train
sampleTraining(model,train_generator,validation_generator,batch_size,epochs)
model.save('model1.h5')
model.save_weights('model_weighte1.h5')

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator, y_pred))
print('Classification Report')
target_names = ['FrenchFries', 'Pizza', 'VegBurger']
print(classification_report(validation_generator, y_pred, target_names=target_names))


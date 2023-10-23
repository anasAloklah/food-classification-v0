from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as k

classify = Sequential()

# optimiz function to minimize the loss & suites for large data & little memo requirement
classify.compile(loss='categorical_crossentropy', optimizer='adam', matrics=['accuracy'])
# CNN
imag = (128, 128, 3)
classify.add(Conv2D(32, (3, 3), activation='relu'))(imag)
classify.add(MaxPooling2D(pool_size=(2, 2)))
classify.add(Conv2D(64, (3, 3), activation='relu'))
classify.add(MaxPooling2D(pool_size=(2, 2)))
classify.add(Conv2D(128, (3, 3), activation='relu'))
classify.add(MaxPooling2D(pool_size=(2, 2)))

classify.add(Flatten())
classify.add(Dense(128, activation='relu'))
classify.add(Dense(3, activation='softmax'))

# train
data_train = ImageDataGenerator( rescale=1. / 255 ,shear_range = 0.2,zoome_range = 0.2, horizontal_flip = True )
data_test = ImageDataGenerator(
    rescale=1. / 255
)
train_set = data_train.flow_from_directory('food_photos', target_size=(128, 128), batch_size=32, class_mod='categorical')
classify.fit_generator(train_set, steps_per_epoch=800 / 32, epochs=50, validation_data=data_test,
                       validation_steps=200 / 32)








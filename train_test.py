
# importing libraries
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


img_width, img_height = 224, 224
#insert dataset
train_data_dir = 'treaning'
validation_data_dir = 'test'
num_of_train_samples = 480+321+550
num_of_test_samples = 12+80+147
epochs = 30
batch_size = 64
num_classes=3
# check format of img if it's bainary it will be add 3 layers
if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)
# initial CNN 
classify = Sequential()
classify.add(Conv2D(32, (3, 3), input_shape = input_shape))
classify.add(Activation('relu'))
classify.add(MaxPooling2D(pool_size =(2, 2)))

classify.add(Conv2D(32, (3, 3)))
classify.add(Activation('relu'))
classify.add(MaxPooling2D(pool_size =(2, 2)))

classify.add(Conv2D(64, (3, 3)))
classify.add(Activation('relu'))
classify.add(MaxPooling2D(pool_size =(2, 2)))


classify.add(Conv2D(128, (3, 3)))
classify.add(Activation('relu'))
classify.add(MaxPooling2D(pool_size =(2, 2)))

classify.add(Flatten())
classify.add(Dense(64))
classify.add(Activation('relu'))
classify.add(Dropout(0.5))#  to avoid overfitting on the dataset.
classify.add(Dense(1))
classify.add(Dense(num_classes, activation='softmax'))
# optimiz function to minimize the loss & suites for large data & little memo requirement
classify.compile(loss ='binary_crossentropy',
					optimizer ='rmsprop',
				metrics =['accuracy'])
# train& test dataset
train_data = ImageDataGenerator(
				rescale = 1. / 255,
				shear_range = 0.2,
				zoom_range = 0.2,
			horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_data.flow_from_directory(train_data_dir,
							target_size =(img_width, img_height),
					batch_size = batch_size, class_mode ='categorical')

validation_generator = test_data.flow_from_directory(
									validation_data_dir,
				target_size =(img_width, img_height),
		batch_size = batch_size, class_mode ='categorical')
train_generator.filenames
classify.fit_generator(train_generator,
	steps_per_epoch = num_train_samples // batch_size,
	epochs = epochs, validation_data = validation_generator,
	validation_steps = num_validation_samples // batch_size)

classify.save('model.h5')
classify.save_weights('model_weighte.h5')



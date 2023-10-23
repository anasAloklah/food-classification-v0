from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
from keras.optimizers import SGD, RMSprop, Adam
from DatasetHandler import load_images_formList
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn.model_selection import train_test_split

img_rows = 224
img_cols = 224
num_classes=3
train_data_path = 'food_photos'


####### Load concatenated data from disk

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
                                                    batch_size=64,
                                                    class_mode='categorical'
                                                    )
####### Create train/val/test split
print("Creating train/val/test/split")
n_classes = num_classes

X_train, X_test, y_train, y_test = train_test_split(train_generator._filepaths,train_generator.classes, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.05, random_state=1)

y_train_cat = to_categorical(y_train,num_classes)
y_test_cat = to_categorical(y_test, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
X_train=load_images_formList(X_train,img_rows,img_cols)
X_test=load_images_formList(X_test,img_rows,img_cols)
X_val=load_images_formList(X_val,img_rows,img_cols)

X_all = None
X_val_test = None
y_val_test = None

print("Writing X_test.hdf5")
h = h5py.File('X_test.hdf5', 'w')
h.create_dataset('data', data=X_test)
h.create_dataset('classes', data=y_test_cat)
h.close()

######## Set up Image Augmentation
print("Setting up ImageDataGenerator")
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    rescale=1./255,
    fill_mode='nearest')
datagen.fit(X_train)
generator = datagen.flow(X_train, y_train_cat, batch_size=32)
val_generator = datagen.flow(X_val, y_val_cat, batch_size=32)


## Fine tuning. 70% with image augmentation.
## 83% with pre processing (14 mins).
## 84.5% with rmsprop/img.aug/dropout
## 86.09% with batchnorm/dropout/img.aug/adam(10)/rmsprop(140)
## InceptionV3

K.clear_session()

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_rows, img_cols, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
# # x = Flatten()(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(.5)(x)
predictions = Dense(n_classes, activation='softmax')(x)

# x = base_model.output
# x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
# x = Flatten(name='flatten')(x)
# predictions = Dense(101, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
print("First pass")
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
csv_logger = CSVLogger('first.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_val_samples=len(y_val),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=10,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

print("Second pass")
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
csv_logger = CSVLogger('second.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_val_samples=len(y_val),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=100,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])
model.save('model1.h5')
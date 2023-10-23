import numpy as np
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
#Start

test_data_path = 'test' # change to target test data set
img_rows = 224
img_cols = 224
batch_size = 16
num_of_test_samples = 347 # change to target number of test data set samples
num_classes=3 # change to target num class
n_classes=num_classes
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
model.load_weights('model_weighte1.h5')
model.save('model1.h5')
test_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

X_train, X_test, y_train, y_test = train_test_split(validation_generator._filepaths,validation_generator.classes, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.05, random_state=1)
y_classes=validation_generator.classes
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_classes, y_pred))
print('Classification Report')
target_names = ['FrenchFries', 'Pizza', 'VegBurger']
print(classification_report(y_classes, y_pred, target_names=target_names))


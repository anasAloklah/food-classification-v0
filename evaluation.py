import numpy as np
from keras.models import load_model
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

model = load_model ('model1.h5')
model.load_weights('model_weighte1.h5')

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


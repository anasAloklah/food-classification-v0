from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # convert img with PIL to numpy array for deep learning
import numpy as np
#import requests # to get data from specific resource

from io import BytesIO # for non text data
from keras.preprocessing import image  # for new img
from PIL import Image , ImageTk # to read data(img) from url
#from tkinter import Tk,Label,Canvas,NW,Entry,Button
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

model = load_model ('model.h5')
model.load_weights('model_weighte.h5')
img_width, img_height = 224, 224


#path_img = '1.jpg'
#res = requests.get(path_img) #to test the request
#test_image = Image.open(BytesIO('1.jpg'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img = image.load_img('1.jpg', target_size=(img_width, img_height),grayscale=False, color_mode='rgb')
#imgplot = plt.imshow(img)
#plt.show()

x = image.img_to_array(img) # convert img into numpy array
x = x.astype('float32')
x /= 255
x = np.expand_dims(x, axis=0)  #expand the shape of array (1,3,224,224) send the img into network in batches for efficincy


image = np.vstack([x])

#test_image=image.load_img('1.jpg' ,grayscale=False, color_mode='rgb',  target_size=(img_width, img_height),    interpolation='nearest')
#test_image = image.load_img('1.jpg',grayscale=False,color_mode='rgb', img_width, img_height) # take path of img from user
#test_image = image.img_to_array(test_image) # convert img into numpy array
#test_image = image_utils.img_to_array(test_image) # the shape of img array is(3,224,244)
#test_image = np.expand_dims(test_image, axis = 0) #expand the shape of array (1,3,224,224) send the img into network in batches for efficincy

prediction = model.predict_classes(image) #predict the result
print (prediction)
"""
if prediction ==[0]:
   res= 'd'
elif prediction ==[1]:
   res='f'
elif prediction ==[2]:
   res='h'
   
elif prediction ==[3]:
    res='pizza'
elif prediction ==[4]:
    res='salad'
elif prediction ==[5]:
    res='spaghetti'
elif prediction ==[6]:
    res='waffle'
print(res)"""

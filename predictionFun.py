from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # convert img with PIL to numpy array for deep learning
import numpy as np
#import requests # to get data from specific resource
import xlrd
from io import BytesIO # for non text data
from keras.preprocessing import image  # for new img
from PIL import Image , ImageTk # to read data(img) from url
#from tkinter import Tk,Label,Canvas,NW,Entry,Button
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
def predcition(imgPath):
    from keras.preprocessing import image  # for new img
    model = load_model ('model.h5')
    model.load_weights('model_weighte.h5')
    img_width, img_height = 224, 224
    #path_img = '1.jpg'
    #res = requests.get(path_img) #to test the request
    #test_image = Image.open(BytesIO('1.jpg'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    img = image.load_img(imgPath, target_size=(img_width, img_height),grayscale=False, color_mode='rgb')
    #imgplot = plt.imshow(img)
    #plt.show()

    x = image.img_to_array(img) # convert img into numpy array
    x = x.astype('float32')
    x /= 255
    x = np.expand_dims(x, axis=0)  #expand the shape of array (1,3,224,224) send the img into network in batches for efficincy


    image = np.vstack([x])
    path = r'food&calor.xlsx'
    b = xlrd.open_workbook(path)
    sheet = b.sheet_by_index(0)
    sheet.cell_value(0, 0)
    prediction = model.predict_classes(image) #predict the result
    print (prediction)
    res=''
    if prediction == [0]:
     res = 'your meal and it is contains  '.join(sheet.row_values(0, 0))
    elif prediction == [1]:
        res = 'your meal and it is contains   '.join(sheet.row_values(1, 0))
    elif prediction == [2]:
        res = 'your meal and it is contains   '.join(sheet.row_values(2, 0))
    elif prediction == [3]:
        res = 'your meal and it is contains   '.join(sheet.row_values(3, 0))
    elif prediction == [4]:
        res = 'your meal and it is contains  '.join(sheet.row_values(4, 0))
    elif prediction == [5]:
        res = 'your meal and it is contains   '.join(sheet.row_values(5, 0))
    return  res

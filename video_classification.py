#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:29:10 2019

@author: Rose Li
"""

import imageio
import keras
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np    # for mathematical operations
from keras.preprocessing import image   # for preprocessing the images
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from keras.utils import np_utils
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from skimage.transform import resize   # for resizing images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import generic_utils
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os
import shutil
import csv

# create csv file
def create_csv(csv_name):
    path = csv_name
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        # save ID, dir and label of the image
        csv_head = ["image_ID","Path","Class","Exist"]
        csv_write.writerow(csv_head)

# write csv file,save ID, dir and label of the image
def write_csv(csv_name,image_id,image_dir,label,mark):
    path  = csv_name
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [image_id,image_dir,label,mark]
        csv_write.writerow(data_row)

# create csv file for videos
def create_csv_video(csv_name):
    path = csv_name
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        # save ID, dir and label of the image
        csv_head = ["image_ID","Path"]
        csv_write.writerow(csv_head)

# write csv file for videos,save ID, dir
def write_csv_video(csv_name,image_id,image_dir):
    path  = csv_name
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [image_id,image_dir]
        csv_write.writerow(data_row)

# create csv file for result
def create_csv_result(csv_name):
    path = csv_name
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        # save ID, dir and label of the image
        csv_head = ["video_id","Class"]
        csv_write.writerow(csv_head)

# write csv file for videos results,save ID, Class
def write_csv_result(csv_name,image_id,video_label):
    path  = csv_name
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [image_id,video_label]
        csv_write.writerow(data_row)
               
# read images in folder
def read_images_in_folder(csv_name,folder):
    for filename in os.listdir(folder):
        # # change label 99 to 10, to reduce the dimension of one hot encoding
        # if int(folder[-2:])!=99:        
        #     write_csv(csv_name,filename,folder,int(folder[-2:]),1)
        # #else:
        #     #write_csv(csv_name,filename,folder,-1,-1)
        write_csv(csv_name,filename,folder,int(folder[-1]),1)

# read image folders
def read_image_folder(csv_name,image_folder):
    for filename in os.listdir(image_folder):
        # determine if it is a folder
        if os.path.isdir(image_folder+filename):
            # if so, read images inside
            read_images_in_folder(csv_name,image_folder+filename)    
            
# read video, split video to frames and save the images to the correspoding imageset
def read_video(csv_name_video,video_name,image_set_name):
    vid = imageio.get_reader(video_name,'ffmpeg')
    count = 0
    for num,im in enumerate(vid):
        if num%40==0:
            imageio.imwrite(image_set_name+'/frame%d.jpg'%count,im)
            write_csv_video(csv_name_video,'frame%d'%count,image_set_name)
            count+=1
        
if __name__== "__main__":        
    # preprocessing the data
    # read the image information and save into a csv file
    csv_name = "data.csv"
    create_csv(csv_name)  
    image_folder = r"./"+"data/image/"
    video_folder = r"./"+"data/input_video/"
    # new_image_folder = r"./"+"data/preview2/" 
    read_image_folder(csv_name,image_folder)
    # read_image_folder(csv_name,new_image_folder)
    
    # reading the csv file
    data = pd.read_csv('data.csv')     
    # data.head()      # printing first five rows of the file
 
    # read images into X
    X = [ ]     
    for index, row in data.iterrows():
        img = plt.imread(row["Path"]+'/'+ row["image_ID"])
        X.append(img)  # storing each image in array X
    X = np.array(X)    # converting list to array
    
    # read labels
    y = data.Class
    # one hot encoding Classes
    y_one_hot = np_utils.to_categorical(y)    
    
    # reshape the images to fit the VGG16
    images = []
    for i in range(0,X.shape[0]):
        # reshape the image
        a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)
        images.append(a)
    X = np.array(images)
    # preprocessing the input data
    X = preprocess_input(X, mode='tf')
    # for cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=12)
    X_t,X_v,y_t,y_v = X_train, X_test, y_train, y_test
    # use imagenet and VGG16 to do the transfer learning,remove the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    X_train = base_model.predict(X_train)
    X_test = base_model.predict(X_test)
    # reshape the model to 1D to fully connected layer
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    X_train = X_train.reshape(X_train_shape[0], X_train_shape[1]*X_train_shape[2]*X_train_shape[3])
    X_test = X_test.reshape(X_test_shape[0], X_test_shape[1]*X_test_shape[2]*X_test_shape[3])
    # centering the data
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max() 
    
    # build the model
    model = Sequential()
    # input layer
    model.add(InputLayer((X_train_shape[1]*X_train_shape[2]*X_train_shape[3],)))    # input layer
    # hidden layer
    model.add(Dense(units=1024, activation=(tf.nn.sigmoid)))
    #model.add(Dense(units=1024, activation=(tf.nn.sigmoid),kernel_regularizer=regularizers.l2(0.5),activity_regularizer=regularizers.l1(0.5)))
    # adding dropout
    model.add(Dropout(0.5))
    #model.add(Dense(units=512, activation=(tf.nn.sigmoid)))    # hidden layer
    #model.add(Dropout(0.5))
    #model.add(Dense(units=256, activation=(tf.nn.sigmoid)))    # hidden layer
    #model.add(Dropout(0.5))    
    # output layer, the number of class is len(y_one_hot[0])
    model.add(Dense(len(y_one_hot[0]), activation=(tf.nn.softmax))) 
    
    # #use GPU to compile the model    
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # #save best weights,handle imbalance data
    # class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)  # computing weights of different classes    
    # filepath="weights.best.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]      # model check pointing based on validation loss
    # model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), class_weight=class_weights, callbacks=callbacks_list,batch_size=32)
    
    model.load_weights("weights.best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),batch_size=32)
    #model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    # save the results
    csv_name_result = "result.csv"
    create_csv_result(csv_name_result)
    # test the video
    # split the video and save in different folder
    for filename in os.listdir(video_folder):
        file = filename.split('.')        
        folder = os.path.exists(file[0])
        #if the folder not exist, create it
        if not folder:            
            os.makedirs(file[0])
        else:
            #clear the files inside
            shutil.rmtree(file[0])
            os.mkdir(file[0])
        csv_name_video = "video"+file[0]+".csv"
        create_csv_video(csv_name_video)
        # split video to images
        read_video(csv_name_video,video_folder+'/'+filename,file[0])
        # test the video images by training model
        test = pd.read_csv(csv_name_video)
        # read images into X
        test_image = [ ]     
        for index, row in test.iterrows():
            img = plt.imread(row["Path"]+'/'+ row["image_ID"]+'.jpg')
            test_image.append(img)  # storing each image in test image array 
        test_img = np.array(test_image)    # converting list to array
        
        test_image = []
        for i in range(0,test_img.shape[0]):
            a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
            test_image.append(a)
        test_image = np.array(test_image)
        
        # preprocessing the images
        test_image = preprocess_input(test_image, mode='tf')
        
        # extracting features from the images using pretrained model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        test_image = base_model.predict(test_image)
        
        # converting the images to 1-D form
        test_image_shape = test_image.shape
        test_image = test_image.reshape(test_image.shape[0], test_image.shape[1]*test_image.shape[2]*test_image.shape[3])
        # zero centered images
        test_image = test_image/test_image.max()
        
        # make predictions for test image
        predictions = model.predict_classes(test_image)
        
        label_video = ""
        max_class = 0
        # calculate the screen time for each class,find the max one except non-object class 10
        for i in range(len(y_one_hot[0])):           
            if predictions[predictions==i].shape[0]>0:
                print("The screen time of %d class is"%i, predictions[predictions==i].shape[0], "seconds")
                #label_video += " "
                #label_video += str(i)
            if predictions[predictions==i].shape[0]>predictions[predictions==max_class].shape[0]:
                max_class = i
        print(filename,max_class)
        label_video = str(max_class)
        # write the final result
        write_csv_result(csv_name_result,filename,label_video)
    
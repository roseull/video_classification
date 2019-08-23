#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:29:10 2019

@author: Xuan Li
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
        # change label 99 to 10, to reduce the dimension of one hot encoding
        if int(folder[-2:])!=99:        
            write_csv(csv_name,filename,folder,int(folder[-2:]),1)
        #else:
            #write_csv(csv_name,filename,folder,-1,-1)

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
    for num,im in enumerate(vid):
        imageio.imwrite(image_set_name+'/frame%d.jpg'%num,im)
        write_csv_video(csv_name_video,'frame%d'%num,image_set_name)
        
if __name__== "__main__":        
    # preprocessing the data
    # read the image information and save into a csv file
    csv_name = "data.csv"
    create_csv(csv_name)  
    image_folder = r"./"+"data/image/" 
    read_image_folder(csv_name,image_folder)
    
    # reading the csv file
    data = pd.read_csv('data.csv')     
    #data.head()      # printing first five rows of the file
    
    # data augmentation
    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            #rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='nearest')
    
    new_image_folder = r"./"+"data/preview2"  
    folder = os.path.exists(new_image_folder)
    #if the folder not exist, create it
    if not folder:            
        os.makedirs(new_image_folder)
    else:
        #clear the files inside
        shutil.rmtree(new_image_folder)
        os.mkdir(new_image_folder)
        
    for index, row in data.iterrows():
        img = load_img(row["Path"]+'/'+ row["image_ID"])
        name = row["Path"].split("/")
        path = new_image_folder+'/'+name[-1]
        folder = os.path.exists(path)
        #if the folder not exist, create it
        if not folder:            
            os.makedirs(path)
            
        x = img_to_array(img)  # this is a Numpy array
        x = x.reshape((1,) + x.shape)  # this is a Numpy array
        i = 0
        # 20 images for each image
        for batch in datagen.flow(x, batch_size=1,save_to_dir = path, save_prefix='new', save_format='jpeg'):
            i += 1
            if i > 15:
                break  # otherwise the generator would loop indefinitely

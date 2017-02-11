# All libraries loaded/ imported here
import argparse
import base64
import json
import numpy as np
import cv2
import os
import skimage.io
import scipy.misc
import PIL
import pandas as pd
import tensorflow as tf
import socketio
import time
import math
import random

from PIL import Image
from PIL import ImageOps

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json

from io import BytesIO

# All variables defined here
epochs = 5
batch_size = 256
test_size = 0.25
Resized_col, Resized_row = 64, 64
input_shape = (Resized_col, Resized_row, 3)
dropout = 0.5
samples_per_epoch = 20 * 1024
null = ''

def augment_image_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def translate_image(image, steer, translate_by_pixel):
    # Translation - shifting images horizonatally/ vertically and adjusting the steering angle
    # Add 0.004 steering angle units per pixel shift to the right, 
    # and subtract 0.004 steering angle units per pixel shift to the left.
    trans_x = translate_by_pixel*np.random.uniform()-translate_by_pixel/2
    steer_ang = steer + trans_x/translate_by_pixel * 2 * 0.2
    trans_y = 10 * np.random.uniform() - 10/2
    trans_M = np.float32([[1,0,trans_x],[0,1,trans_y]])
    image_tr = cv2.warpAffine(image,trans_M,(cols,rows))    
    return image_tr,steer_ang,trans_x

def resize_image(image):
    # Remove the top and bottom 25% and bottom 25 pixels and then resize it to 64x 64
    shape = image.shape
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(Resized_col,Resized_row), interpolation=cv2.INTER_AREA)    
    return image

def choose_camera(selected_data):
    
    camera_sel = np.random.randint(3)
    
    # Check for empty camera data.. 
    camera_check = type(selected_data['right'][0]) # Could have been 'right' as well
    if not (camera_check == str ) or camera_check == null:        
        camera_sel = 0  # Use Center Camera Data if no left/ right camera present
        
    if (camera_sel == 0): # Center Camera Selected
        camera_data = selected_data['center'][0].strip()
        shift_ang = 0.0
    if (camera_sel == 1): # Left Camera Selected
        camera_data = selected_data['left'][0].strip()
        shift_ang = 0.25  # Adjust steering angle 
    if (camera_sel == 2): # Right Camera Selected
        camera_data = selected_data['right'][0].strip()
        shift_ang = -0.25  # Adjust steering angle 
    
    y_steer = selected_data['steer'][0] + shift_ang
    
    return camera_data, y_steer

def preprocess_images(selected_data):
    # Selects camera, translates images (horizontally or vertically), augments
    # brightness for all images and flips about half the images
    camera_data, y_steer = choose_camera(selected_data)
    image = cv2.imread(camera_data)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image, y_steer, trans_x = translate_image(image, y_steer, 100)
    image = augment_image_brightness(image)
    image = resize_image(image)
    image = np.array(image)
    
    # flip probability 50% 
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        y_steer = -y_steer
    
    return image, y_steer

def generate_train_valid_data_from_batch(data, batch_size = 32):
    # Data generator for keras training, for both training and validation  
    X_val = np.zeros((batch_size, Resized_row, Resized_col, 3))
    y_val = np.zeros(batch_size)
    
    while 1:
        for i in range(batch_size):
            selected_sample = np.random.randint(len(data))
            selected_data = data.iloc[[selected_sample]].reset_index()
            x, y = preprocess_images(selected_data)
            
            # Remove samples with angles closer to 0 @ 50% probability to avoid overfitting
            # Straight line drives
            keep = 0
            while keep == 0:
                x, y = preprocess_images(selected_data)
                if abs(y) < 0.15:
                    if random.random() < 0.5:
                        keep = 1
                else:
                    keep = 1
            
            X_val[i] = x
            y_val[i] = y
        yield X_val, y_val

def my_model(input_shape):
    
    # NVIDIA CNN architecture using Keras/ Tensorflow with modification by Vivek Yadav
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.83ddq78y9
    
    model = Sequential()
    
    # Normalization
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape, name='Normalization'))

    # Convolutional layers with Maxpooling and dropout to prevent overfitting        
    model.add(Convolution2D(3, 1, 1, border_mode='same', subsample=(2, 2), activation='elu'))
    
    # 1st Set of convoloutions/ maxpooling and dropout
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Dropout(dropout))
    #
    # 2nd Set of convoloutions/ maxpooling and dropout
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Dropout(dropout))
    #
    # 3rd Set of convoloutions/ maxpooling and dropout
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Dropout(dropout))
    #
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1, activation='elu'))
    
    # Adam optimization defined with learning rate = 0.0001   
    adam = Adam(lr=1e-4)
    
    model.compile(optimizer="adam", loss="mse")
    
    return model

# Reads the data Files, sets up and calls the model generator

if __name__ == '__main__':
    
    print ("Reading the CSV file into PD array..")
    driving_log = 'driving_log.csv'
    drive_log_array = pd.read_csv(driving_log, index_col = False)  
    
    print ("drive_log_array size", np.shape(drive_log_array))
    
    drive_log_array.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    
    image = cv2.imread(drive_log_array['center'][0].strip())
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    rows, cols, channels = image.shape
    
    model = my_model(input_shape)    
    model.summary()
    
    # Set the validation sample size 
     
    nb_val_samples = samples_per_epoch * test_size
    
    print ("Samples per Epoch..", samples_per_epoch) 
    print ("Validation Samples..", nb_val_samples)  
      
    data_generator = generate_train_valid_data_from_batch(drive_log_array, batch_size)
    
    model.fit_generator(data_generator, nb_epoch=epochs, 
                        samples_per_epoch = samples_per_epoch, nb_val_samples = nb_val_samples)
    
 
    # Save model to JSON file    
    print ("Saving model and weights files..")
    model.save_weights('model.h5')
    json = model.to_json()
    with open('model.json', 'w') as out:
        out.write(json)



import os
from os.path import join,exists
from tensorboard import program
import datetime
import numpy as np
import pandas as pd
import cv2
import pickle
from tqdm import tqdm
from PIL import Image
from random import randint
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import convert_to_tensor
import tensorflow.keras.backend as K
from tensorflow.keras.models import  load_model,Model
from tensorflow.keras import applications
from tensorflow.keras.layers import Lambda,Input,Dropout, Activation, Dense, GlobalMaxPooling2D,Conv2D,Flatten,MaxPooling2D,InputLayer
from tensorflow.keras.utils import to_categorical
from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras import datasets

from adv_benchmark.config import Config




def pick_data_set(name):

    if name=='Mnist':
        data_mnist=datasets.mnist.load_data(path='mnist.npz')
        X_train,y_train=data_mnist[0][0],data_mnist[0][1]
        X_test,y_test=data_mnist[1][0],data_mnist[1][1]
        y_train = to_categorical(y_train_mnist, 10)
        y_test = to_categorical(y_test_mnist, 10)

        X_train = np.full((60000, 32, 32, 3), 0)
        for i, s in enumerate(X_train):
            X_train[i] = cv2.cvtColor(np.pad(s,2), cv2.COLOR_GRAY2RGB) 
            
        X_test = np.full((10000, 32, 32, 3), 0)
        for i, s in enumerate(X_test):
            X_test[i] = cv2.cvtColor(np.pad(s,2), cv2.COLOR_GRAY2RGB) 
        
    
    elif name=='Cifar':
        (X_train, y_train), (X_test, y_test), _,_=load_dataset('cifar10')
        for i, im in enumerate(X_train):
            X_train[i]=255*im
        for i, im in enumerate(X_test):
            X_test[i]=255*im 

    return(X_train,X_test,y_train,y_test)





def train_and_save_effnet(data_set_name):
    (X_train,X_test,y_train,y_test)=pick_data_set(data_set_name)
    tf.keras.backend.clear_session()
    effnet_base = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    effnet_base.trainable=True
    x = GlobalMaxPooling2D(name='pool_1')(effnet_base.layers[-2].output)
    x = Dropout(0.2, name="dropout_2")(x)
    x = Dense(32)(x)
    x = Dense(10,name='fc_2')(x)
    o = Activation('softmax', name='act_2')(x)
    model_effnet = Model(inputs=effnet_base.input, outputs=[o])


    
    if exists(Config.MODELS_PATH+'/effnet_model_'+str(data_set_name)+'.h5')==False:
        model_effnet.compile(
            loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=['accuracy']
            )
        history = model_effnet.fit(X_train, y_train,
                      epochs=5,
                      batch_size = 128,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=1)
        model_effnet.save(Config.MODELS_PATH+'/effnet_model_'+str(data_set_name)+'.h5')

    else:
        model_effnet=load_model(Config.MODELS_PATH+'/effnet_model_'+str(data_set_name)+'.h5')
        
    return(model_effnet)

def train_and_save_small_model(data_set_name):
    
    if exists(Config.MODELS_PATH+'/small_model_'+str(data_set_name)+'.h5')==False:
        (X_train,X_test,y_train,y_test)=pick_data_set(data_set_name)
        tf.keras.backend.clear_session()   
        small_model = tf.keras.models.Sequential()
        small_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32,32,3)))
        small_model.add(MaxPooling2D(2, 2))
        small_model.add(Conv2D(64, (3, 3), activation='relu'))
        small_model.add(MaxPooling2D(2, 2))
        small_model.add(Flatten())
        small_model.add(Dense(128, activation='relu'))
        small_model.add(Dense(10, activation='softmax'))


        small_model.compile(
            loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=['accuracy']
            )


        history = small_model.fit(X_train, y_train,
                      epochs=10,
                      batch_size =128,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=1)


        small_model.save(Config.MODELS_PATH+'/small_model_'+str(data_set_name)+'.h5')
    else:
        small_model=load_model(Config.MODELS_PATH+'/small_model_'+str(data_set_name)+'.h5')
    return(small_model)
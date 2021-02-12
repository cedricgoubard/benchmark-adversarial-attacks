'''
Module for implementing the random padding and resizing defense strategy
'''

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
from foolbox.attacks import LinfFastGradientAttack,LinfDeepFoolAttack
from foolbox.distances import LpDistance
from foolbox.models import TensorFlowModel
from foolbox import criteria
from sklearn.metrics import classification_report,plot_roc_curve

from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import BoundaryAttack
from art.utils import load_dataset #to play with cifar images
from tensorflow.keras.layers.experimental import preprocessing

from adv_benchmark.config import Config
from adv_benchmark.models_training import pick_data_set
from adv_benchmark.metrics import DOC,succes_rate
from adv_benchmark.models_training import train_and_save_effnet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4*1024)]
)

tf.config.run_functions_eagerly(True) # this otpion is required to make the random padding works (otherwise it is not random anymore)


class ResizePad(tf.keras.layers.Layer):
    '''
    define a new tensorflow layer that takes and image (32 by 32) and randomly resize it (less than 32*32)
    and randomly pad it back to a 32 by 32 image 
    '''
    def __init__(self):
        super(ResizePad,self).__init__()
       
        
    def resize_and_pad(self,image):
        new_size=randint(13,27)
        left_offset=randint(0,32-new_size)
        top_offset=randint(0,32-new_size)
        image=tf.cast(image,dtype='float32')
        paddings = tf.constant([[0,0],[top_offset,32-top_offset-new_size],[left_offset,32-left_offset-new_size],[0,0]])
        image=tf.image.resize(image, [new_size,new_size], method='nearest', preserve_aspect_ratio=True)
        image=tf.pad(image, paddings, mode='CONSTANT', constant_values=0, name=None)
        return(image)
    
    def call(self, x):
        return self.resize_and_pad(x)



def create_model_with_defense(data_set_name):
    '''
    input: 
    -data set name (str) : name of the data set on which to train the model ('Mnist' or 'Cifar' )
    -output: a trained tensorflow model
    '''

    (X_train,X_test,y_train,y_test)=pick_data_set(data_set_name)
    model_without_def=train_and_save_effnet(data_set_name)
    
    
    model_with_def = tf.keras.models.Sequential()
    model_with_def.add(Input(shape=(32,32,3)))
    model_with_def.add(ResizePad())
    model_with_def.add(model_without_def)


    model_with_def.compile(
        loss='categorical_crossentropy',
        optimizer='nadam',
        metrics=['accuracy']
        )
    if exists(Config.MODELS_PATH+'/random_padding/'+str(data_set_name)+'.h5')==False:
        history = model_with_def.fit(X_train, y_train,
                      epochs=5,
                      batch_size = 32,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=1)
        model_with_def.save_weights(Config.MODELS_PATH+'/random_padding/'+str(data_set_name)+'.h5')
    else:
        model_with_def.load_weights(Config.MODELS_PATH+'/random_padding/'+str(data_set_name)+'.h5')
          
    
    return(model_with_def)
'''
Module for implementing the denoising defense strategy
'''

import os
from os.path import join,exists


import numpy as np
import pandas as pd
import cv2
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random

import tensorflow as tf

from tensorflow import convert_to_tensor
import tensorflow.keras.backend as K
from tensorflow.keras.models import  load_model,Model
from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2DTranspose, Reshape,Input,Dropout, Activation, Dense, GlobalMaxPooling2D,Conv2D,Flatten,MaxPooling2D,InputLayer
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

from adv_benchmark.config import Config
from adv_benchmark.models_training import pick_data_set
from adv_benchmark.metrics import DOC,succes_rate


class Denoise(Model):
    '''
    This class creates a autoencoder tensorflow model 
    which takes an image as input and returns the same image wihtout the potential adversarial noise
    '''
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          Input(shape=(32, 32, 3)), 
          Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
          Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
          Flatten(),
          Dense(4096,activation='relu'),          
          Reshape((8,8,64))
            ])

        self.decoder = tf.keras.Sequential([
        Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(128, kernel_size=2, strides=2, activation='relu', padding='same'),
        Conv2D(3, kernel_size=(3,3), activation='relu', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def data_set_maker(model,attack, image_list, labels):
    '''
    This function creates the data set that will be used to train the autoencoder (denoiser)
    This data set is made of couples of adversarial/bening images
    inputs:
    -model (tensorflow model): the model that will be attacked to produced the adversarial examples
    -attack (foolbox attack): attack that will produce the adversarial images
    -image_list (list of arrays): images that will be attacked 
    -labels (list on one hot encoded labels): labels of the image 
    output:
    -adv_list (list of numpy arrays): adversarial images
    -benign_list (list of numpy arrays): benign images corresponding to the images in adv_list 
    -adv_true_label: true labels of the images (attention here the are not one hot encoded)
    
    '''
    model_to_fool = TensorFlowModel(model, bounds=(0, 255))
    success=[]
    adv_list=[]
    benign_list=[]
    adv_true_label=[]
    epsilon=[5]
    labels=list(map(np.argmax,labels))
    print('======epsilon: '+str(epsilon[0])+'======')
    for i,image in enumerate(tqdm(image_list,position=0)):
        if i!=0 and i%(len(labels)//3)==0:
            print('======adv_list_size: '+str(len(adv_list))+'======')
            epsilon=[epsilon[0]*1.5]
            print('======epsilon: '+str(epsilon[0])+'======')
        image = np.asarray(image)[:,:,:3].astype('float32')
        image = convert_to_tensor(np.expand_dims(image,axis=0))
        label=labels[i]
        label = tf.convert_to_tensor(np.array([label]))
        _, clipped, is_adv = attack(model_to_fool,image,label,epsilons=epsilon)
        if bool(is_adv[0]) ==True:
            adv_list.append(np.array(clipped[0][0]))
            adv_true_label.append(labels[i]) 
            benign_list.append(image) 
            
    for i,image in enumerate(benign_list):
        benign_list[i]=np.squeeze(image)
        
    return(list(adv_list),list(benign_list),adv_true_label)


def make_adv_data_set(data_set_name,model_effnet,number_of_image_to_use=6000,attack=LinfFastGradientAttack()):
    '''
    This function creates the data set that will be used to train the autoencoder (same than previous function)
    but it also add some couples (benign image, bening image) so that the model 'understands' that 
    not all images are adversarial. It also shuffles it. 
    inputs:
    -data_set_name: 'Cifar' or 'Mnist'
    -model_effnet (tensorflow model): model that will be attacked to produced the adv examples
    -number_of_image_to_use (int): number of images used to produced adv examples (the resulting number of adv images will be less than that)
    -attack (foolbox attack): attack used to produce the adversarial example
    outputs:
    -adv_list (list of numpy arrays): adversarial images
    -benign_list (list of numpy arrays): benign images corresponding to the images in adv_list 
    -adv_true_label: true labels of the images (attention here the are not one hot encoded)

  
    '''
    
    (X_train,X_test,y_train,y_test)=pick_data_set(data_set_name)
    if exists(Config.DATA_PATH+'adv images and benign images '+str(data_set_name))==False:
        (adv_list,benign_list,adv_true_label)=data_set_maker(model_effnet,attack, X_test[:number_of_image_to_use], y_test[:number_of_image_to_use])
        with open(Config.DATA_PATH+'adv images and benign images '+str(data_set_name), 'wb') as f:
            pickle.Pickler(f).dump(adv_list)
            pickle.Pickler(f).dump(benign_list)
            pickle.Pickler(f).dump(adv_true_label)
    else:
        with open(Config.DATA_PATH+'adv images and benign images '+str(data_set_name), 'rb') as f:
            adv_list=pickle.Unpickler(f).load()
            benign_list=pickle.Unpickler(f).load()
            adv_true_label=pickle.Unpickler(f).load()

    ###let's add some benign examples to the data set and shuffle the result

    adv_list.extend(X_test[number_of_image_to_use:number_of_image_to_use+1000])
    benign_list.extend(X_test[number_of_image_to_use:number_of_image_to_use+1000])
    adv_true_label.extend(list(map(np.argmax,y_test[number_of_image_to_use:number_of_image_to_use+1000])))

    adv_list=np.array(adv_list)
    benign_list=np.array(benign_list)
    adv_true_label=np.array(adv_true_label)

    indices = np.arange(len(adv_list))
    random.shuffle(indices)
    adv_list = adv_list[indices]
    benign_list = benign_list[indices]
    adv_true_label = adv_true_label[indices]
    return(adv_list,benign_list,adv_true_label)
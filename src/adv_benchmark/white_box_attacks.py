'''
Module for running white box attacks and computing their performances
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

import tensorflow as tf

from tensorflow import convert_to_tensor
import tensorflow.keras.backend as K
from tensorflow.keras.models import  load_model,Model
from tensorflow.keras import applications
from tensorflow.keras.layers import Dropout, Activation, Dense, GlobalMaxPooling2D,Conv2D,Flatten,MaxPooling2D,InputLayer
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

from adv_benchmark.metrics import DOC,succes_rate



def attack_performances_computer(model_to_attack,predicting_model,attack, image_list, labels,epsilon):
    '''
    This fonction launch an attack against a model and returns the performances of the attack
    inputs:
    -model_to_attack (tensorflow model): model that will be attacked 
    -predicting_model (tensorflow model): model that will predict the label of the generated 
                adv example (most of the time it is the same that model_to_attack) but sometimes
                it is usefull to have another model taking care of the prediction
    -attack (foolbox attack)
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon (float): epsilon is the amount of noise added into the image at each step

    outputs:
    -DOC_attack (float) : average degreee of change of the attack 
    -SR_on_attacked_model (float): success rate of the attack on the first model of the inputs (model_to_attack) 
    -SR_on_predicting_model (float) : success rate of the attack on the second model of the inputs (predicting_model) 
    '''
    
    model_to_attack=TensorFlowModel(model_to_attack , bounds=(0, 255))
    success_on_attacked_model=[]
    success_on_predicting_model=[]
    adv_list=[]
    labels=list(map(np.argmax,labels))
    for i,image in enumerate(tqdm(image_list,position=0)):
        image = np.asarray(image)[:,:,:3].astype('float32')
        image = convert_to_tensor(np.expand_dims(image,axis=0))
        label=labels[i]
        label = tf.convert_to_tensor(np.array([label]))
        _, clipped, is_adv = attack(model_to_attack,image,label,epsilons=epsilon)
        success_on_attacked_model.append(bool(is_adv[0]))
        adv_list.append(np.array(clipped[0]))
        prediction=predicting_model.predict(clipped[0])
        if np.argmax(prediction)!=labels[i]:
            success_on_predicting_model.append(True)
        else:
            success_on_predicting_model.append(False)
    DOC_attack=DOC(adv_list,image_list)
    SR_on_attacked_model=succes_rate(success_on_attacked_model)
    SR_on_predicting_model=succes_rate(success_on_predicting_model)
    return(DOC_attack,SR_on_attacked_model,SR_on_predicting_model)


def attack_runner(model_to_attack,predicting_model,image_list, labels_list, epislons_list,attack=LinfFastGradientAttack()):  
    '''
    This fonction launch an attack against a model and returns the performances of the attack
    
    -model_to_attack (tensorflow model): model that will be attacked 
    -predicting_model (tensorflow model): model that will predict the label of the generated 
    adv example (most of the time it is the same that model_to_attack) but sometimes
    it is usefull to have another model taking care of the prediction
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon_list (list of floats): list of epsilon to try 

    outputs:
    -SR_on_attacked_model (dict): result of the attack on model to attack - keys: average DOC values computed for one epsilon, values: average success rate (SR) values computed for one epsilon
    -SR_on_predicting_model (dict):result of the attack on predicting_model - keys: average DOC values computed for one epsilon, values: average success rate (SR) values computed for one epsilon
    '''
    
    SR_on_attacked_model={}  
    SR_on_predicting_model={}
    for epsilon in epislons_list:
        print('======'+'epislon: '+str(epsilon)+'======')
        DOC,SR_attacked,SR_pred=attack_performances_computer(model_to_attack,predicting_model,attack, image_list, labels_list,[epsilon])      
        SR_on_attacked_model[DOC]=SR_attacked
        SR_on_predicting_model[DOC]=SR_pred  
    return(SR_on_attacked_model,SR_on_predicting_model)


def three_attacks_runner(model_to_attack,predicting_model,image_list, labels_list, epislons_list):
    '''
    This fonction launch three attacks (FGSM, deepfool 1 step and deepfool mutliple steps)
    against a model and returns the performances of the attacks
    
    -model_to_attack (tensorflow model): model that will be attacked 
    -predicting_model (tensorflow model): model that will predict the label of the generated 
    adv example (most of the time it is the same that model_to_attack) but sometimes
    it is usefull to have another model taking care of the prediction
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon_list (list of floats): list of epsilon to try 

    outputs:
    -SR_deepfool_1step_dic (dict): result of the attack on model to attack - keys: average DOC values computed for one epsilon, 
    values: average success rate (SR) values computed for one epsilon for FGSM
    -SR_deepfool_mutliple_steps_dic (dict): result of the attack on model to attack - keys: average DOC values computed for one epsilon, 
    values: average success rate (SR) values computed for one epsilon for deepfool with 1 step
    -SR_FGSM_dic (dict): result of the attack on model to attack - keys: average DOC values computed for one epsilon, 
    values: average success rate (SR) values computed for one epsilon for deepfool 

    '''
    attack_deepfool_1_steps=LinfDeepFoolAttack(steps=1)
    attack_deepfool_mutliple_steps=LinfDeepFoolAttack()
    attack_FGSM=LinfFastGradientAttack()
    
    if len(labels_list[0])<=5:
        labels_list=list(map(np.argmax,labels_list))

    SR_FGSM_dic={}
    SR_deepfool_1step_dic={}
    SR_deepfool_mutliple_steps_dic={}


    for epsilon in epislons_list:
        print('======'+'epislon: '+str(epsilon)+'======')
        DOC_FGSM,SR_FGSM,_=attack_performances_computer(model_to_attack,predicting_model,attack_FGSM, image_list, labels_list,[epsilon])
        DOC_deepfool_1step,SR_deepfool_1step,_=attack_performances_computer(model_to_attack,predicting_model,attack_deepfool_1_steps, image_list, labels_list,[epsilon])
        DOC_deepfool_mutliple_steps,SR_deepfool_mutliple_steps,_=attack_performances_computer(model_to_attack,predicting_model,attack_deepfool_mutliple_steps, image_list, labels_list,[epsilon])


        SR_deepfool_1step_dic[DOC_deepfool_1step]=SR_deepfool_1step   
        SR_deepfool_mutliple_steps_dic[DOC_deepfool_mutliple_steps]=SR_deepfool_mutliple_steps    
        SR_FGSM_dic[DOC_FGSM]=SR_FGSM
        
    return(SR_deepfool_1step_dic,SR_deepfool_mutliple_steps_dic,SR_FGSM_dic)
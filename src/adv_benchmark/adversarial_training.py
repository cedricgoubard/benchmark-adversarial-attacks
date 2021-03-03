"""
Module for implementing the adversarial training defense strategy
"""
from os.path import exists, join
import time
import itertools

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dropout,
    Activation,
    Dense,
    GlobalMaxPooling2D,
)
from efficientnet.tfkeras import EfficientNetB7

from adv_benchmark.config import get_cfg
from adv_benchmark.utils import compute_acc
from adv_benchmark.models_training import pick_data_set


class NewModel(tf.keras.Model):
    '''
    Defines a new tensorflow model with a custom objective function that implements the 
    adversarial training
    This Model keeps the efficientNet architecture
    '''
    def __init__(self,learning_rate,epsilon):
        '''        
        inputs:
        -learning rate (float): learning rate of the Nadam optimizer
        -epsilon (float): amount of added noise in the adversarial objective function
        '''
        super(NewModel,self).__init__()
        
        self.c=0
        self.learning_rate=learning_rate
        self.epsilon=epsilon
        
        
        self.effnet_base = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        self.effnet_base.trainable=True
        x=self.effnet_base.layers[-2].output        
        x=GlobalMaxPooling2D()(x)
        x=Dropout(0.2)(x)
        x=Dense(32)(x)
        x=Dense(10)(x)
        o=Activation('softmax')(x)
        self.model= Model(inputs=self.effnet_base.inputs, outputs=[o])


        
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Nadam(self.learning_rate)#self.learning_rate)        
        
   
    @tf.function
    def train_step(self,x,y):
        '''
        computes custom objective function, computes gradient and applies it
        inputs: 
        -x (tensor): an image
        -y (label tensor): label one hot encoded
        
        '''
        
        with tf.GradientTape() as tape_w:  
            tape_w.watch(self.model.trainable_variables)
            
            with tf.GradientTape() as tape_x:
                tape_x.watch(x)
                true_loss = self.loss(y, self.model(x))        
            gradient = tape_x.gradient(true_loss, x)
            signed_grad = tf.sign(gradient) 
            adv_image=tf.clip_by_value(x+self.epsilon*signed_grad,0,255)
            adv_loss= self.loss(y, self.model(adv_image)) 
           
       
      # Combines both losses            
            total_loss= (1-self.c)*true_loss + self.c*adv_loss
            
    # Regular backward pass.
            gradients = tape_w.gradient(total_loss, self.model.trainable_variables)   
            
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
       
    def train(self,epochs=5,c_list=[0,0.1,0.3,0.5,0.7,0.9],data_set_list=['Cifar','Mnist']):
        '''
        This methods trains several model and save their weigths. This avoids to instanciate several model
        which leads to OOM errors. It saves len(c_list)*len(data_set_list) models
        inputs: 
        -epochs(int>0): number of training epochs
        -c_list (list of float between 0 and 1): contains the c parameters for the objective function 0=no adv training, 1=adv training only
        -data_set_list (list of str): names of the data sets on which to train the model
        
        '''
                
        for data_set_name in data_set_list:
            
            for c in c_list:
                (X_train,X_test,y_train,y_test)=pick_data_set(data_set_name)
                train_data_set=tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(buffer_size=100000).batch(128)
                print("======= training on data_set: "+str(data_set_name)+' c:'+str(c)+'======')
                self.c=c
                if exists(Config.MODELS_PATH+'adversarial_training/'+str(data_set_name)+'/c='+str(c)+'.h5')==False:
                    pred=list(map(np.argmax,self.model(X_test[:1000])))
                    true_values=list(map(np.argmax,y_test[:1000]))
                    acc=np.sum([1 for i in range(len(pred)) if pred[i]==true_values[i]])/len(pred)         
                    print('Accuracy before training is {} '.format(acc))
                    print('-----------')
                    for epoch in range(epochs):                        
                        start = time.time()
                        for (x,y) in tqdm(train_data_set,position=0):
                            x=tf.cast(x,dtype='float32')
                            self.train_step(x,y)           

                        pred=list(map(np.argmax,self.model(X_test[:1000])))
                        true_values=list(map(np.argmax,y_test[:1000]))
                        acc=np.sum([1 for i in range(len(pred)) if pred[i]==true_values[i]])/len(pred)         

                        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                        print('-----------')
                        print('Accuracy for epoch {} is {} '.format(epoch + 1, acc))   
                        
                    self.model.save_weights(Config.MODELS_PATH+'adversarial_training/'+str(data_set_name)+'/c='+str(c)+'.h5')
                else:
                    self.model.load_weights(Config.MODELS_PATH+'adversarial_training/'+str(data_set_name)+'/c='+str(c)+'.h5')
                    pred=list(map(np.argmax,self.model(X_test[:1000])))
                    true_values=list(map(np.argmax,y_test[:1000]))
                    acc=np.sum([1 for i in range(len(pred)) if pred[i]==true_values[i]])/len(pred)         

                    print('-----------')
                    print('Accuracy is {} '.format(acc))   
                    
        return()
    
    def call(self, inputs):
        x = self.model(inputs)
        return x


def train_models():
    '''
    Instanciates one model with custom loss and train them 
    outputs:
    -model_to_load_weights_onto (tensorflow model) this will be used to load the weights that have been saved
    during training 
    '''
   
    new_model=NewModel(learning_rate=0.0001,epsilon=5)
    new_model.train(epochs=5)
    
    model_to_load_weights_onto=new_model.model
    model_to_load_weights_onto.compile(
            loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=['accuracy']
            )
           
    return(model_to_load_weights_onto)   
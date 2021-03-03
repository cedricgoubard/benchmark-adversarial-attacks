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
        """
        cfg = get_cfg()

        for data_set_name, c_param in itertools.product(data_set_list, c_list):
            save_path = join(
                cfg.MODELS_PATH, "adversarial_training", str(data_set_name), f"c={c_param}.h5"
                )

            (X_train, X_test, y_train, y_test) = pick_data_set(data_set_name)
            train_data_set = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(buffer_size=100000)
                .batch(128)
            )

            print(f"\n======= Looking for dataset: {data_set_name} c: {c_param} =======")
            self.c_param = c_param

            if not exists(save_path):
                print(f"Dataset not found at {save_path}; training new model")
                acc = compute_acc(self.model, X_test[:1000], y_test[:1000])

                print(f"Accuracy before training is {acc}\n--------------")
                for epoch in range(epochs):
                    start = time.time()
                    for (data, label) in tqdm(train_data_set, position=0):
                        data = tf.cast(data, dtype="float32")
                        self.train_step(data, label)

                    acc = compute_acc(self.model, X_test[:1000], y_test[:1000])

                    print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")
                    print("-----------")
                    print(f"Accuracy for epoch {epoch + 1} is {acc} ")

                self.model.save_weights(save_path)

            else:
                print(f"Dataset found at {save_path}; using pretrained model")
                self.model.load_weights(save_path)
                acc = compute_acc(self.model, X_test[:1000], y_test[:1000])

                print("-----------")
                print(f"Accuracy is {acc}")

        return ()

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
"""
Module for implementing the random padding and resizing defense strategy
"""
from os.path import exists
from random import randint

import tensorflow as tf
from tensorflow.keras.layers import Input

from adv_benchmark.config import get_cfg
from adv_benchmark.models_training import pick_data_set
from adv_benchmark.models_training import train_and_save_effnet


gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4 * 1024)]
)

tf.config.run_functions_eagerly(
    True
)  # this otpion is required to make the random padding works (otherwise it is not random anymore)


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
    """

    (X_train, _, y_train, _) = pick_data_set(data_set_name)
    model_without_def = train_and_save_effnet(data_set_name)

    cfg = get_cfg()
    model_path = cfg.MODELS_PATH + "/random_padding/" + str(data_set_name) + ".h5"

    model_with_def = tf.keras.models.Sequential()
    model_with_def.add(Input(shape=(32,32,3)))
    model_with_def.add(ResizePad())
    model_with_def.add(model_without_def)


    model_with_def.compile(
        loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
    )
    if not exists(model_path):
        _ = model_with_def.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
        )
        model_with_def.save_weights(model_path)
    else:
        model_with_def.load_weights(model_path)

    return model_with_def

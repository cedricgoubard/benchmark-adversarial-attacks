"""
Module training and saving models
"""
from os.path import exists

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dropout,
    Activation,
    Dense,
    GlobalMaxPooling2D,
    Conv2D,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from efficientnet.tfkeras import EfficientNetB7
from art.utils import load_dataset  # to play with cifar images

from adv_benchmark.config import get_cfg


def pick_data_set(name):
    """
    This fonction return a training and testing set
    input:
    -name : 'Cifar' or 'Mnist'
    output:
    -X_train: list of 60000 32*32*3 images
    -X_test: list of 10000 32*32*3 images
    -y_train: list of 50000 one hot encoded vectors
    -y_test: list of 10000 one hot encoded vectors

    """

    if name.lower() == "mnist":
        data_mnist = datasets.mnist.load_data(path="mnist.npz")
        X_train_mnist, y_train = data_mnist[0][0], data_mnist[0][1]  # pylint: disable=invalid-name
        X_test_mnist, y_test = data_mnist[1][0], data_mnist[1][1]  ## pylint: disable=invalid-name
        y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

        X_train = np.full((60000, 32, 32, 3), 0)
        X_test = np.full((10000, 32, 32, 3), 0)

        prep_func = lambda img: cv2.cvtColor(np.pad(img, 2), cv2.COLOR_GRAY2RGB)

        for i, img in enumerate(X_train_mnist):
            X_train[i] = prep_func(img)

        for i, img in enumerate(X_test_mnist):
            X_test[i] = prep_func(img)

    elif name == "Cifar":
        (X_train, y_train), (X_test, y_test), _, _ = load_dataset("cifar10")
        for i, img in enumerate(X_train):
            X_train[i] = 255 * img
        for i, img in enumerate(X_test):
            X_test[i] = 255 * img

    return (X_train, X_test, y_train, y_test)


def train_and_save_effnet(data_set_name):
    """
    This fonction train (or load) and save an instance of EfficientNet
    -name : 'Cifar' or 'Mnist'
    output:
    -model_effnet (tensorflow model): trained instance of EfficientNet

    """
    (X_train, _, y_train, _) = pick_data_set(data_set_name)
    tf.keras.backend.clear_session()
    effnet_base = EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(32, 32, 3)
    )
    effnet_base.trainable = True
    layer = GlobalMaxPooling2D(name="pool_1")(effnet_base.layers[-2].output)
    layer = Dropout(0.2, name="dropout_2")(layer)
    layer = Dense(32)(layer)
    layer = Dense(10, name="fc_2")(layer)
    output = Activation("softmax", name="act_2")(layer)
    model_effnet = Model(inputs=effnet_base.input, outputs=[output])

    cfg = get_cfg()
    model_path = cfg.MODELS_PATH + "effnet_model_" + str(data_set_name) + ".h5"
    if not exists(model_path):
        model_effnet.compile(
            loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
        )
        _ = model_effnet.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
        )
        model_effnet.save(model_path)

    else:
        model_effnet = load_model(model_path)

    return model_effnet


def train_and_save_small_model(data_set_name):
    """
    This fonction train (or load) and save an instance of a small custom CNN
    -name : 'Cifar' or 'Mnist'
    output:
    -small model (tensorflow model): trained instance of the small model

    """
    cfg = get_cfg()
    model_path = cfg.MODELS_PATH + "small_model_" + str(data_set_name) + ".h5"
    if not exists(model_path):
        (X_train, _, y_train, _) = pick_data_set(data_set_name)
        tf.keras.backend.clear_session()
        small_model = tf.keras.models.Sequential()
        small_model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        small_model.add(MaxPooling2D(2, 2))
        small_model.add(Conv2D(64, (3, 3), activation="relu"))
        small_model.add(MaxPooling2D(2, 2))
        small_model.add(Flatten())
        small_model.add(Dense(128, activation="relu"))
        small_model.add(Dense(10, activation="softmax"))

        small_model.compile(
            loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
        )

        _ = small_model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
        )

        small_model.save(model_path)
    else:
        small_model = load_model(model_path)
    return small_model

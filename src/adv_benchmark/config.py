"""
config module
-------------

This module enables the user to change some main features of the project
"""

from os import makedirs, getenv
from os.path import join


class Config(object):
    '''
    MODELS_PATH : where the models are saved
    DATA_PATH: where data are saved
    '''
    MODELS_PATH = "/media/hdd1/adversarial_attacks_benchmark/models/"
    DATA_PATH = "/media/hdd1/adversarial_attacks_benchmark/data/"

    CIFAR_CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    
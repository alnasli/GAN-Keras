from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import  Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


def build_discriminator(img_shape):
    """
    Build the discriminator
    :param img_shape: input shape of the discriminator
    :return: discriminator, keras model
    """
    img=Input(shape=img_shape)
    X=Flatten(input_shape=img_shape)(img)
    X=Dense(512)(X)
    X=LeakyReLU(alpha=0.2)(X)
    X=Dense(256)(X)
    X=LeakyReLU(alpha=0.2)(X)
    validity=Dense(1, activation='sigmoid')(X)

    discriminator=Model(img,validity)
    discriminator.summary()
    return discriminator
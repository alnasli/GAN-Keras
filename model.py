from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import  Model
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

def build_generator(latent_dim,img_shape):
    """
    Build the generator
    :param latent_dim: dimension of the input noise
    :param img_shape:
    :return: generator model
    """

    noise=Input(shape=(latent_dim,)) #generated noise

    #GAN layers
    X=Dense(256, input_dim=latent_dim)(noise)
    X=LeakyReLU(alpha=0.2)(X)
    X=BatchNormalization(momentum=0.8)(X)
    X=Dense(512)(X)
    X=LeakyReLU(alpha=0.2)(X)
    X=BatchNormalization(momentum=0.8)(X)
    X=Dense(1024)(X)
    X=LeakyReLU(alpha=0.2)(X)
    X=BatchNormalization(momentum=0.8)(X)
    X=Dense(np.prod(img_shape), activation='tanh')(X)

    img=Reshape(img_shape)(X) #output of the generator
    generator=Model(noise,img)
    generator.summary()
    return generator

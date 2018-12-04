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

def sample_images(epoch, latent_dim, generator):
        """
        Plot some results during the training.
        """

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images_2/%d.png" % epoch)
        plt.close()
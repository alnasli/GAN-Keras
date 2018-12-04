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
        
def train(epochs=30000, batch_size=32, sample_interval=1000,latent_dim=100):

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    img_shape=X_train.shape[1:]

    #make the combined model of gan
    #input of the combine model
    noise=Input(shape=(latent_dim,))

    # Build and compile the discriminator
    discriminator = build_discriminator(img_shape)
    optimizer = Adam(lr=0.0002)
    discriminator.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
    # Build the generator
    generator = build_generator(latent_dim,img_shape)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The generator takes noise as input and generates imgs
    img=generator(noise)

    # The discriminator takes generated images as input and determines validity
    validity=discriminator(img)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined=Model(noise,validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size,latent_dim))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        if epoch % 50 == 0:
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(epoch,latent_dim,generator)

from keras.datasets import mnist
from keras.layers import Input
from keras.models import  Model
from keras.optimizers import Adam
from model import build_generator,build_discriminator
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--epochs', type=int, nargs='?', default=30000,
                    help='max epoch')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=32,
                    help='set batch size')
parser.add_argument('-l', '--latent_dim', type=int, nargs='?', default=100,
                    help='set latent dim')
parser.add_argument('--lr', type=float, nargs='?', default=0.0002,
                    help='set learning rate')
args = parser.parse_args()

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
    fig.savefig("images/%d.png" % epoch)
    plt.close()

def train(epochs=30000, batch_size=32,learning_rate=0.0002,latent_dim=100, sample_interval=1000):

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
    optimizer = Adam(lr=learning_rate)
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

    generator.save('generator_weights.h5')
    combined.save('combined_weights.h5')
    print("Weights saved")
if __name__ == '__main__':
    epochs=args.epochs
    batch_size=args.batch_size
    latent_dim=args.latent_dim
    learning_rate=args.lr
    train(epochs,batch_size,learning_rate,latent_dim)

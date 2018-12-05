from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

def sample_images(latent_dim, generator):
    """
    Test the image results.
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
    plt.show()
    plt.close()

if __name__ == '__main__':
    generator=load_model('generator_weights.h5')
    latent_dim=100

    sample_images(latent_dim,generator)

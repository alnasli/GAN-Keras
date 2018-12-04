from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import  Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

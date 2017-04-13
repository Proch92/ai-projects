import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

plt.imshow(x_train[0])
plt.show()

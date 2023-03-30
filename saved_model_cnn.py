import tensorflow.keras.models
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype(np.float32)

X_test = X_test.astype(np.float32)
# print(X_train)

X_train = X_train / 255
# y_train = to_categorical(y_train)
X_test = X_test / 255

model = tensorflow.keras.models.load_model('mymodel')

#model.fit(X_train, y_train, batch_size=32, epochs=10)
model.save('mymodel')

print(model.evaluate(X_test, y_test))

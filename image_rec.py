import numpy as np
import matplotlib.pyplot as plt
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
#


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(len(np.unique(y_train)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=50)

print(model.evaluate(X_test, y_test))
model.save('mymodel')

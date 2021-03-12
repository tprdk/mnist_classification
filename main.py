import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#set log level to only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#params
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'x_train shape : {x_train.shape} - y_train shape : {y_train.shape}')
print(f'x_test shape : {x_test.shape} - y_test shape : {y_test.shape}')

image_row    = x_train.shape[1]
image_column = x_train.shape[2]


x_train = x_train.reshape(-1, image_row * image_column).astype('float32') / 255.0
x_test  = x_test.reshape(-1, image_row * image_column).astype('float32') / 255.0

print(f'x_train shape : {x_train.shape} - y_train shape : {y_train.shape}')
print(f'x_test shape : {x_test.shape} - y_test shape : {y_test.shape}')


model = keras.Sequential()
model.add(keras.Input(shape=image_row * image_column))
model.add(layers.Dense(512, activation='sigmoid'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)
model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=True)
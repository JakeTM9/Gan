# https://www.tensorflow.org/tutorials/keras/classification
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

##What is this Data? -> Ill print out some stuff for you

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

##Data Processeing

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

##scales so vals fall between 0 and 1

train_images = train_images / 255.0

test_images = test_images / 255.0

##1st 25 images from training set with class names

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

## Building the Model

##Layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), ## transform from 2d array to 1d array 28x28 -> 784
    keras.layers.Dense(128, activation='relu'), ##2 dense layers are densley connected or fully connected nuerel layers -> 1st has 128 nodes
    keras.layers.Dense(10) ## The second returns a logits array with length 10
])


##Compiling the Model

model.compile(optimizer='adam', ## how the model is updated
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ## how accuratley the model is doing
              metrics=['accuracy']) ##monitor the training and testing steps

##train model
model.fit(train_images, train_labels, epochs=10)

##evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)



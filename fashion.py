#!/usr/bin/env python3

import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


train_images, train_labels = load_mnist('.', kind='train')
test_images, test_labels = load_mnist('.', kind='t10k')


train_images = train_images / 255.0
test_images = test_images / 255.0


network = keras.Sequential([
    keras.layers.Dense(300, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dense(80, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

network.compile(optimizer=keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=3)

preds = network.predict(test_images)
results = []

for p in preds:
    best = 0
    besti = -1
    for i in range(10):
        if p[i] > best:
            best = p[i]
            besti = i
    results.append(besti)
    
matches = 0
for i in range(len(results)):
    if results[i] == test_labels[i]:
        matches += 1

print("Accuracy: %d %%" % (matches/len(results)*100))

network.save("fashion_network.h5")

#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

network = keras.models.load_model('fashion_network.h5')

im = Image.open("triko.png")
imarr = np.array(im)
arr2 = np.reshape(im, (1, 784))
arr2 = arr2 / 255.

preds = network.predict(arr2)

best = 0
besti = 0
for i in range(10):
    if preds[0][i] > best:
        best = preds[0][i]
        besti = i

print("Muj typ je: %s" % class_names[besti])

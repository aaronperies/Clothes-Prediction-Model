from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf            #importing tensorflow
from tensorflow import keras as ks #import keras neural network library

#import numpy as np
#import matplotlib.pyplot as plt

fashion_Dataset = ks.datasets.fashion_mnist                                             #importing the fashion dataset from the keras datasets
(train_images, train_labels), (test_images, test_labels) = fashion_Dataset.load_data()  #structure the datasets in the format that they are imported in
#first set is to train the model so that it learns and 2nd set is to test if it learned correctly

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#list containing the categories of clothes for the images to be classified under

train_images = train_images / 255.0 #scale down the pixels before adding them into the network
test_images = test_images / 255.0

# =============================================================================
# plt.figure(figsize=(10,10))         
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
# =============================================================================

model = ks.Sequential([     #setting up the layers for the model
     ks.layers.Flatten(input_shape=(28, 28)),   #reformats the dataset
     ks.layers.Dense(128, activation='relu'),   #128 nodes
     ks.layers.Dense(10)                        #returns an array with a length of 10
 ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15)

# =============================================================================
# print(len(train_labels))
# print(len(test_labels))
# print(tf.__version__) #prints the tensorflow version installed on the machine
# =============================================================================

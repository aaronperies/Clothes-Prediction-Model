from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf            #importing tensorflow
from tensorflow import keras as ks #import keras neural network library

import numpy as np
import matplotlib.pyplot as plt

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

model.fit(train_images, train_labels, epochs=15) #epochs increased to increase accuracy for the model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]
np.argmax(predictions[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  

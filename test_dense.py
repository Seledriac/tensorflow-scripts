
# -*- coding:utf-8 -*-

#Pt.1


#importing tensorflow
import tensorflow as tf
import numpy as np
#importing pyplot, our visualization library
from matplotlib import pyplot as plt

#importing the mnist dataset from keras
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_train, axis=1).reshape(x_test.shape[0], -1)

#Building the model as sequential (= feedforward)
model = tf.keras.models.Sequential()
#One flatten layer to turn images from 28x28 to 784x1 ndarrays
model.add(tf.keras.layers.Flatten())
#Dense layers (hidden regular layers, connected to each prior and subsequent node)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#Output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model for it to 'fit' the training data
model.fit(x_train, y_train, epochs=3)

#Evaluating the model on the test data
model.evaluate(x_test, y_test)

#Saving the model in a model file
model.save('test_model.model')

#Deserializing the model
new_model = tf.keras.models.load_model('test_model.model')

#And checking the performances with a prediction
predictions = new_model.predict(x_test)
#The first element of the x_test data is a seven digit
print(np.argmax(predictions[0]))
plt.imshow(x_test[0], cmap='Greys')
plt.show()


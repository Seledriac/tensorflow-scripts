# This script creates a convolutional model with keras, trains it to make it fit the mnist dataset, and displays the loss and accuracy graphs

# libraries
import tensorflow as tf
from matplotlib import pyplot as plt

# getting mnist 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# creating the model
model = tf.keras.models.Sequential()

# The input has to be reshaped to fit as a two-dimensional matrix
model.add(tf.keras.layers.Reshape((28, 28, 1)))

# Convolution + pooling layer 
# 64 3x3 filters, relu activation function
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(28,28,1), activation=tf.nn.relu))
# 2x2 pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# Another conv + pooling
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(28,28,1), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# We flatten the result and feed it into a dense layer (64 hidden neurons)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))

# The output layer is a dense layer of 10 neurons, for digit classification
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# The model is compiled with an appropriate loss and optimizer
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Training of the model to fit the mnist data
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

print(history.history.keys())


# Accuracy and loss plotting

plt.figure(1)

plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()

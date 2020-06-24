
#Test images predictions (handwritten digits)

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

#Serialized model retrieving
loaded_model = tf.keras.models.load_model('test_model.model')
loaded_model.evaluate(x_test, y_test)

#Custom Image files loading
examples = []
import os
for filename in next(os.walk("custom_test_images"))[2]:
    examples.append(255 - np.array(Image.open("custom_test_images/" + filename)))

#Predicting for each custom_test_image
examples = np.array(examples)
examples_normalized = tf.keras.utils.normalize(examples, axis=1).reshape(np.array(examples).shape[0], -1)
predictions = loaded_model.predict(examples_normalized)
for ex,pred in zip(enumerate(examples_normalized), predictions):
    print("Example number : {}, Predicted number : {}".format(ex[0], np.argmax(pred)))
    plt.imshow(examples[ex[0]], cmap='Greys')
    plt.show()

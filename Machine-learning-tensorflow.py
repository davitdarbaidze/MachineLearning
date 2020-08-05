import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split







def generate_dataset(numb_sample,test_size):


    x = np.array([[random()/2 for _ in range(2)] for _ in range(numb_sample)])
    y = np.array([[i[0] + i[1]] for i in x])

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_size)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = generate_dataset(5000,0.3)
#print("x_test : \n{}".format(x_test))
#print("y_test : \n{}".format(y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5,input_dim = 2, activation = "sigmoid"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

optimiser = tf.keras.optimizer.SGD(learning_rate = 0.1)
model.compile(optimize = optimiser, loss = " MSE ")

model.fit(x_train,y_train,epochs = 100)
import pandas as pd
from os import system, name
from time import sleep
import timeit
import random
import numpy as np
from tabulate import tabulate
import pdfkit as pdf
import tensorflow as tf
from tensorflow import keras


def clear():
    _ = system('clear')


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def get_tranq_data():
    ''' Reads in data from tranq output files for match prediction
    '''

    import pickle as pkl
    import numpy as np
    trainX = np.array(pkl.load( open("trainX.pkl", "rb" ) ))
    trainY = np.array(pkl.load( open("trainY.pkl", "rb" ) ))

    return trainX, trainY


def main():
    #train_X, test_X, train_y, test_y = get_iris_data()


    train_X, train_y = get_tranq_data()

    model = keras.Sequential()

    model.add(Dense()


    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 8 features and no bias
    h_size = 50                 # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (team 1 wins or team 2 wins)


    clear()
    for epoch in range(100000):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        '''
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))
        '''

        print("Epoch = %d, train accuracy = %.2f%%" #, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy)) #, 100. * test_accuracy))
    sess.close()


if __name__ == '__main__':
    main()

import pandas as pd
from os import system, name
from time import sleep
import timeit
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split


def clear():
    _ = system('clear')

# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2



RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat
'''
def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    from sklearn import datasets
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
'''
def get_tranq_data():
    '''
        Reads in data from tranq output files for match prediction
    '''
    '''
    import csv
    dataFile = open("trainX.csv")
    trainX = dataFile.readlines()
    dataFile = open("trainY.csv")
    trainY = dataFile.readlines()
    '''
    import pickle as pkl
    import numpy as np
    trainX = np.array(pkl.load( open("trainX.pkl", "rb" ) ))
    trainY = np.array(pkl.load( open("trainY.pkl", "rb" ) ))
    testX = np.array(pkl.load( open("testX.pkl", "rb" ) ))
    testY = np.array(pkl.load( open("testY.pkl", "rb" ) ))

    return trainX, trainY, testX, testY


def main():
    #train_X, test_X, train_y, test_y = get_iris_data()
    clear()

    trainX, trainY, testX, testY = get_tranq_data()


    # Layer's sizes
    x_size = trainX.shape[1]   # Number of input nodes: 8 features and no bias
    h_size = 50                 # Number of hidden nodes
    y_size = trainY.shape[1]   # Number of outcomes (team 1 wins or team 2 wins)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD

    sess=tf.Session()    
    #First let's load meta graph and restore weights
    '''
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('savedWeights/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('savedWeights/./'))
        sess.run(predict, feed_dict={X: testX, y: testY})
    '''

    '''
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict ={w1:13.0,w2:17.0}
    
    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
    print sess.run(op_to_restore,feed_dict)
    '''

    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    clear()

    for epoch in range(2000):
        # Train with each example
        for i in range(len(trainX)):
            sess.run(updates, feed_dict={X: trainX[i: i + 1], y: trainY[i: i + 1]})

        train_accuracy = np.mean(np.argmax(trainY, axis=1) ==
                                 sess.run(predict, feed_dict={X: trainX, y: trainY}))

        if epoch % 100 == 0:
            print("Epoch = %d, train accuracy = %.2f%%" #, test accuracy = %.2f%%"
                % (epoch + 1, 100. * train_accuracy)) #, 100. * test_accuracy))
    
    saver.save(sess, 'savedWeights/model')

    print(len(testX))
    print(sess.run(predict, feed_dict={X: testX}))
    
    sess.close()


if __name__ == '__main__':
    main()
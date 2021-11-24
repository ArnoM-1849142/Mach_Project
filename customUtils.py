# used for manipulating directory paths
import os
# Scientific and vector computation for python
import numpy as np
# math library 
import math
# Plotting library
from matplotlib import pyplot
# Optimization module in scipy
from scipy import optimize
# Used for imorting csv data
import csv
# utilies library from the exercises of the machine learning course
import utils

def importImageDataFromCSV(csv_path, data_size=6500, width=50, height=50):
    
    X = np.zeros((data_size, height*width*3))
    y = np.zeros((data_size,13))
    alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M']

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            X[i,:] = row[1:]
            # find the position of the label in the alphabet
            j = alpha.index(row[0][0])
            # set the label to 1 at that position
            y[i,j] = 1
            i += 1
    return X, y

def sigmoidGradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z. 
    This should work regardless if z is a matrix or a vector. 
    In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    
    Parameters
    ----------
    z : array_like
        A vector or matrix as input to the sigmoid function. 
    
    Returns
    --------
    g : array_like
        Gradient of the sigmoid function. Has the same shape as z. 
    
    
    """

    g = np.zeros(z.shape)

    g = utils.sigmoid(z)*(1-utils.sigmoid(z))

    return g

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer_size : int
        Number of hidden units in the second layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,num_labels).
    
    lambda_ : float, optional
        Regularization parameter.
 
    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.
    
    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.
    
    Note 
    ----
    We have provided an implementation for the sigmoid function in the file 
    `utils.py` accompanying this assignment.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1, Theta2 = retrieveThetas(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # Setup some useful variables
    m = y.shape[0]
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # part 1: feedforward   
    # adding column of 1's to X (bias terms)
    a1 = np.concatenate([np.ones((m,1)),X], axis=1)
    
    z2 = np.dot(a1, Theta1.T)
    a2 = utils.sigmoid(z2)
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    z3 = np.dot(a2, Theta2.T)
    h = utils.sigmoid(z3)
                         
    J = 1/m * np.sum(-y*np.log(h)-(1-y)*np.log(1-h))
    # add regularization
    J = J + (lambda_ / (2*m)) * (np.sum(Theta1[:, 1:]**2) +np.sum(Theta2[:, 1:]**2))
    
    # part 2: backpropagation
    delta3 = h - y
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoidGradient(z2)
    
    DELTA1 = np.dot(delta2.T, a1)
    DELTA2 = np.dot(delta3.T, a2)
    
    
    Theta1_grad = 1/m * DELTA1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    Theta2_grad = 1/m * DELTA2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    
    L_out : int
        Number of outgoing connections. 
    
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
    """

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W

def nnCostFunction4L(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a 4 Layer layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer_size : int
        Number of hidden units in the second layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 

    num_layers : int
        Total number of layers (input layer + # hidden layers + output layer)
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,num_labels).
    
    lambda_ : float, optional
        Regularization parameter.
 
    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.
    
    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.
    
    Note 
    ----
    We have provided an implementation for the sigmoid function in the file 
    `utils.py` accompanying this assignment.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 4 layer neural network
    Theta1, Theta2, Theta3 = retrieveThetas4L(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # Setup some useful variables
    m = y.shape[0]
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    Theta3_grad = np.zeros(Theta3.shape)

    # part 1: feedforward   
    # adding column of 1's to X (bias terms)
    a1 = np.concatenate([np.ones((m,1)),X], axis=1)
    
    z2 = np.dot(a1, Theta1.T)
    a2 = utils.sigmoid(z2)
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    z3 = np.dot(a2, Theta2.T)
    a3 = utils.sigmoid(z3)
    a3 = np.concatenate([np.ones((a3.shape[0], 1)), a3], axis=1)

    z4 = np.dot(a3, Theta3.T)
    h = utils.sigmoid(z4)

                         
    J = 1/m * np.sum(-y*np.log(h)-(1-y)*np.log(1-h))
    # add regularization
    J = J + (lambda_ / (2*m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2) + np.sum(Theta3[:, 1:]**2))
    
    # part 2: backpropagation
    delta4 = h - y
    delta3 = np.dot(delta4, Theta3)[:, 1:] * sigmoidGradient(z3)
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoidGradient(z2)
    
    DELTA1 = np.dot(delta2.T, a1)
    DELTA2 = np.dot(delta3.T, a2)
    DELTA3 = np.dot(delta4.T, a3)
    
    Theta1_grad = 1/m * DELTA1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    Theta2_grad = 1/m * DELTA2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    Theta3_grad = 1/m * DELTA3
    Theta3_grad[:, 1:] = Theta3_grad[:, 1:] + (lambda_ / m) * Theta3[:, 1:]

    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel(), Theta3_grad.ravel()])

    return J, grad

def predict4L(Theta1, Theta2, Theta3, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta3.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = utils.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = utils.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    h3 = utils.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h2], axis=1), Theta3.T))
    p = np.argmax(h3, axis=1)
    return p

def retrieveThetas(nn_params, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    return Theta1, Theta2

def retrieveThetas4L(nn_params, input_layer_size, hidden_layer_size, num_labels):
    preceding_values = 0    #stores the amount of values from nn_params that belong to the previous Theta

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    preceding_values += hidden_layer_size * (input_layer_size + 1) 

    Theta2 = np.reshape(nn_params[preceding_values:(preceding_values + hidden_layer_size*(hidden_layer_size + 1))],
                        (hidden_layer_size, (hidden_layer_size + 1)))
    preceding_values += hidden_layer_size*(hidden_layer_size + 1)

    Theta3 = np.reshape(nn_params[preceding_values:],
                        (num_labels, (hidden_layer_size + 1)))
    return Theta1, Theta2, Theta3

def thetasToCSV(filePath, Theta1, Theta2):
    with open(filePath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(Theta1.shape[0]):
            writer.writerow(Theta1[i,:])
        for i in range(Theta2.shape[0]):
            writer.writerow(Theta2[i,:])

def thetasFromCSV(filePath, theta1shape, theta2shape):
    #parameters Theta1shape and Theta2shape indicate the shape of the respective theta matrixes
    Theta1 = np.zeros(theta1shape)
    Theta2 = np.zeros(theta2shape)

    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i in range(theta1shape[0]):
            Theta1[i,:] = next(reader)
        for i in range(theta2shape[0]):
            Theta2[i,:] = next(reader)
    
    return Theta1, Theta2

def thetasToCSV_4L(filePath, Theta1, Theta2, Theta3):
    with open(filePath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(Theta1.shape[0]):
            writer.writerow(Theta1[i,:])
        for i in range(Theta2.shape[0]):
            writer.writerow(Theta2[i,:])
        for i in range(Theta3.shape[0]):
            writer.writerow(Theta3[i,:])

def thetasFromCSV_4L(filePath, theta1shape, theta2shape, theta3shape):
    #parameters thetaXshape indicate the shape of the respective theta matrixes
    Theta1 = np.zeros(theta1shape)
    Theta2 = np.zeros(theta2shape)
    Theta3 = np.zeros(theta3shape)

    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i in range(theta1shape[0]):
            Theta1[i,:] = next(reader)
        for i in range(theta2shape[0]):
            Theta2[i,:] = next(reader)
        for i in range(theta3shape[0]):
            Theta3[i,:] = next(reader)
    
    return Theta1, Theta2, Theta3


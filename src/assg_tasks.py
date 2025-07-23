import math
import numpy as np


def basic_sigmoid(x):
    """Compute sigmoid of the scalar (float/double usualy) input
    parameter and return.  This basic version should not be vectorized,
    and is required to use the `exp` function from the `math` library.

    Arguments
    ---------
    x - a scalar (float/double) value

    Returns
    -------
    scalar float/double - The sigmoid of the input scalar parameter x is calculated and
        returned.
    """
    # your implementation of task 1.1 goes here, don't forget to return the
    # correct result instead of the stub/default answer of 0.0 here
    return 0.0

def sigmoid(x):
    """Compute sigmoid of the input parameter x and return. In this version
    the input parameter might be a scalar, but it could be a list or
    a numpy array.  Your implementation should be vectorized and able to
    hanld all of these.

    Arguments
    ---------
    x - a scalar, python list or numpy array of real valued (float/double) numbers.

    Returns
    -------
    s - Result will be of the same shape as the input and will be the element wise
      calculation of the sigmoid for all values given as input.
    """
    # your implementation of task 1.1 goes here
    return np.array([0])

def sigmoid_grad(x):
    """Compute the gradient (also called the slope or derivative) of the sigmoid function with respect
    to its input x.  You can store the output of the sigmoid function into a variable and then use it to
    calculate the gradient.  You are required to reuse your previous `sigmoid()` function in this function.

    Arguments
    ---------
    x - A scalar or numpy array of real valued numbers

    Returns
    -------
    ds - The computed gradients of the sigmoid of x with respect to the x inputs.
    
    """
    # your implementaiton of task 1.2 goes here
    return np.array([0])

def standard_scalar(x):
    """Implement the standard feature scaling operation on the input array
    x of data.  Each column/feature of the data is centered by subtracting
    the feature mean from the column, and normalized by dividing by the feature
    standard deviation.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (n, m), e.g. n rows x m columns

    Returns
    -------
    x_scaled - The normalized data after standard feature scaling has been applied to
      each feature column. 
    mu - Returns a vector of the means of each feature column, vector should have shape
      (m,)
    sigma - Returns a vector of the standard deviation of each original feature column.
      The vector also has shape (m,)
    """
    # your implementation of task 1.3 goes here
    return np.array([0]), np.array([0]), np.array([0])

def softmax(x):
    """Calculates the softmax for each row of the input 2-D tensor matrix x.
    Your code should work for a row vector of shape (1, n) but also for general
    matrices of shape (m, n).  You shouldn't have to do anything special to handl
    a row vector, numpy broadcasting should work in either case.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (m, n)

    Returns
    -------
    s - A numpy matrix with the same shape (m, n) as the input argument x, with the computed softmax of x
    """
    # your implementation of task 1.4 goes here
    return np.array([0])

def one_hot(category):
    """One hot encode a numpy array of string categorical data.  This function
    expects a simple array/vector as input of shape (n,) with n samples of strings
    that are unique categories.  This function returns a (n,m) shaped one-hot
    encoded matrix, where the number of columns m equals the number of unique
    category strings in the input catagory array.

    Arguments
    ---------
    category - A vector of a single column of data of shape (n,).  This functione expects object
      encoded strings in the inputs

    Returns
    -------
    one_hot_array - Returns an array of shape (n, m) for the n input samples where m is the number of
      unique categories found in the input.
    """
    # your implementation of task 1.5 goes here
    return np.array([0])

def mae(y_pred, y_true):
    """This function computes the L1 loss.  We expect 2 vectors of the
    same size as input to this function.

    Arguments
    ---------
    y_pred - a 1-d tensor / vector of size m, the predicted labels
    y_true - a 1-d tensor / vector of the same size m, the true labels

    Returns
    -------
    loss - the value of the L1 (absolute value) loss function for y_pred and y_true
    """
    # your implementation of task 2.1 goes here
    return 0.0

def rmse(y_pred, y_true):
    """This function computes the mean squared error / L2 loss function.  We expect 2 vectors of the
    same size as input to this function.

    Arguments
    ---------
    y_pred - a 1-d tensor / vector of size m, the predicted labels
    y_true - a 1-d tensor / vector of the same size m, the true labels

    Returns
    -------
    loss - the value of the L2 / MSE loss function for y_pred and y_true
    """
    # your implementation of task 2.2 goes here
    return 0.0

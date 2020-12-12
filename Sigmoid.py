#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1/(1+np.exp(-(x)))
    #raise NotImplementedError
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    x=np.log(s/(1-s))
    ds = np.exp(-(x))/((1+np.exp(-(x))) ** 2)
    #raise NotImplementedError
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print( "Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print (f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print (g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print ("You should verify these results by hand!\n")


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE
    x = np.array([[3, 4, 9], [0, 1, -1]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    f_ans = np.array([
        [0.9525741268224334, 0.9820137900379085,0.9998766054240137],
        [0.5,0.7310585786300049,0.2689414213699951]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print(g)
    g_ans = np.array([
        [0.045176659730912144,0.017662706213291114,0.0001233793497648489],
        [0.25,0.19661193324148188,0.19661193324148185]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
#    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()

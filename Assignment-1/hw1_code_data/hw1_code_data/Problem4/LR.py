# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import math

def sigmoid(z):
    "Returns Sigmoid of the input value"
    return (1/(1+math.exp(-z)));

def cost(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta)) # predicted probability of label 1
    log_l = (-y)*numpy.log(p_1) - (1-y)*numpy.log(1-p_1) # log-likelihood vector

    return log_l.mean()

def grad(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = numpy.dot(error, X_1) / y.size # gradient vector

    return grad

def predict(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta))
    return p_1 > 0.5

import scipy.optimize as opt
from pylab import scatter, show, legend, xlabel, ylabel

#load the dataset
data = numpy.loadtxt('train.txt', delimiter='   ')

X = data[:, 0:57]
y = data[:, 57]

pos = numpy.where(y == 1)
neg = numpy.where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()


theta = 0.1* numpy.random.randn(3)
X_1 = numpy.append( numpy.ones((X.shape[0], 1)), X, axis=1)

theta_1 = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X_1, y))
p = predict(array(theta), it)
#print 'Train Accuracy: %f' % ((y[numpy.where(p == y)].size / float(y.size)) * 100.0)


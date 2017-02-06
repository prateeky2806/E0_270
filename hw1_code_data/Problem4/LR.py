# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy
import pandas as pd
import scipy.optimize as optim


def sigmoid(z):
	"Returns Sigmoid of the input matrix"
	return (1/(1+numpy.exp(-z)));

def costFunction(theta, X, y):
	m=numpy.size(y)
	H = sigmoid(numpy.dot(X_train,theta))
	cost = (1/m)*(-(numpy.dot(numpy.transpose(Y_train),numpy.log(H)))-(numpy.dot(numpy.transpose(1-Y_train),numpy.log(1-H))))
	return cost

def gradient(theta, X, y):
	m= y.shape[0]
	grad = numpy.zeros(numpy.shape(theta))
	H = sigmoid(numpy.dot(X_train,theta))
	grad = (1/m)*(numpy.dot(numpy.transpose(X_train), (H-Y_train)))
	return grad

def predict(theta, X):
	#print(X.shape)
	#print(theta.shape)
	z = X.dot(theta)
	#print(z)
	p_1 = sigmoid(z)
	return p_1 > 0.5

train = numpy.array(pd.read_csv("train.csv", header=0))
test = numpy.array(pd.read_csv("test.csv", header=0))
#print(train)

X_train= train[:,0:-1]
Y_train= train[:,-1]
X_test= test[:,0:-1]
Y_test= test[:,-1]
#print(X_train.shape)
[m,n]=numpy.shape(X_train)
#print(m)
#print(X_test.shape)
#print(n+1)
X_train = numpy.append( numpy.ones((m, 1)), X_train, axis=1)
X_test = numpy.append( numpy.ones((X_test.shape[0], 1)), X_test, axis=1)
#print(X_test[1:20,:])
theta = numpy.zeros((n+1,1))
#cost = costFunction(theta,X_train, Y_train)
#grads = gradient(theta,X_train,Y_train)
#final_theta = optim.minimize(fun = costFunction, x0=theta, args = (X_train, Y_train), method = 'BFGS', jac = gradient)
final_theta = optim.fmin_bfgs(costFunction, theta, fprime = gradient, args = (X_train, Y_train))
#print(final_theta)

p = predict(final_theta,X_test)
print(p)
print(Y_test[numpy.where(p == Y_test)])
print ('Train Accuracy: %f', ((Y_test[numpy.where(p == Y_test)].size / float(Y_test.size)) * 100.0))
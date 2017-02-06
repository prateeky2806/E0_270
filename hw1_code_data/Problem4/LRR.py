# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:26:40 2017

@author: Prateek Yadav
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
from numpy import where
import pandas as pd
import scipy.optimize as optim

def sigmoid(z):
	#Z=numpy.zeros((X_train.dot(theta).shape))
	"Returns Sigmoid of the input value"
	Z= (1/(1+numpy.exp(-z)))
	return Z

#sigmoid=numpy.vectorize(sigmoid, otypes=[numpy.float])

def costFunctionReg(theta, X, y, lam):
	m=numpy.size(y)
	cost=0
	H = sigmoid(numpy.dot(X_train,theta))
	theta0=theta.copy()
	theta0[0]=0
	cost = (1/m)*((-(numpy.dot(numpy.transpose(Y_train),numpy.log(H)))-(numpy.dot(numpy.transpose(1-Y_train),numpy.log(1-H))))+((lam/2)*(theta0.T.dot(theta0))))
	return cost

def gradientReg(theta, X, y, lam):
	m= y.shape[0]
	grad = numpy.zeros(numpy.shape(theta))
	H = sigmoid(numpy.dot(X_train,theta))
	theta0=theta.copy()
	theta0[0]=0
	grad = (1/m)*(numpy.dot(numpy.transpose(X_train), (H-Y_train))+(lam*theta0))
	return grad

def predict(theta, X, y):
	p_1 = sigmoid(X.dot(theta))
	return p_1 > 0.5


train = numpy.array(pd.read_csv("train.csv", header=0))
test = numpy.array(pd.read_csv("test.csv", header=0))
#print(train)

X_train= train[:,0:57]
Y_train= train[:,57]
X_test= test[:,0:57]
Y_test= test[:,57]
#print(X_train)
print("checkpoint-1")
[m,n]=numpy.shape(X_train)
X_train = numpy.append( numpy.ones((m, 1)), X_train, axis=1)
theta = numpy.zeros((n+1,1))
lam = 10**(-7)
print("checkpoint-2")
cost = costFunctionReg(theta,X_train, Y_train, lam)
print("checkpoint-3")
grads = gradientReg(theta,X_train,Y_train, lam)
print("checkpoint-4")
final_theta = optim.minimize(fun = costFunctionReg, x0=theta, args = (X_train, Y_train, lam), method = 'TNC', jac = gradientReg)
print("checkpoint-5")
print(final_theta)
print("checkpoint-6")

p = predict(final_theta,X_test, Y_test)
print ('Train Accuracy: %f', ((Y_test[numpy.where(p == Y_test)].size / float(Y_test.size)) * 100.0))
print("checkpoint-6")
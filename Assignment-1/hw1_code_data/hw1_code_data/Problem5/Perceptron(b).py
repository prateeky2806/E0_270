# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:36:24 2017

@author: Prateek Yadav
"""
import numpy
from numpy import where
import pandas as pd

def Model(X_train, X_test):
	
	return

def Predict()


train_small = numpy.array(pd.read_csv("train_small.csv", header=0))
print(train_small.shape)
X_train_small = train_small[:,0:-1]
Y_train_small = train_small[:,-1]
#print(X_train_small[1:20, -1])
#print(Y_train_small[1:20])
(W, b) = Model(X_train_small, Y_train_small)   # Training Model to obtain Weights and bais

test = numpy.array(pd.read_csv("test.csv", header=0))
X_test = test[:,0:-1]
Y_test = test[:,-1]
print(X_test.shape)
p = Predict(W,b,X_test, Y_test)
print ('Train Accuracy: %f', ((Y_test[numpy.where(p == Y_test)].size / float(Y_test.size)) * 100.0))

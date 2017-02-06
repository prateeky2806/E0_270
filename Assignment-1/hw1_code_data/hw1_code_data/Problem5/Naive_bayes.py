# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:50:50 2017

@author: Prateek Yadav
"""
import numpy
import pandas as pd

def Model(X_train, Y_train):
	
	Y1 = (Y_train==1)
	Y0 = (Y_train==-1)
	X_pos = X_train[Y1]  # training data with positive sentiment
	X_neg = X_train[Y0]  # training data with negative sentiment
	#print(X_pos[1:20])
	#print(X_pos.shape)
	Prob_pos = (sum(X_pos)/(X_pos.shape[0]))   #each column denotes a feature 'x'; and contains P(X=x|Y=1)
	Prob_neg = (sum(X_neg)/(X_neg.shape[0]))   #each column denotes a feature 'x'; and contains P(X=x|Y=-1)
	#print(Prob_pos.size == Prob_neg.size)
	#print(Freq_pos[1:20])
	PY1 = (X_pos.shape[0]/(X_pos.shape[0]+X_neg.shape[0]))  #P(Y=1)
	PY0 = (X_neg.shape[0]/(X_pos.shape[0]+X_neg.shape[0]))  #P(Y=-1)
	Intercept1 = (numpy.log((1-Prob_pos)/(1-Prob_neg))).sum()  # term sigma_over_all_features(log((1-p_i)/(1-q_i)))
	#print(Intercept1.shape)
	Intercept2 = numpy.log(PY1/PY0)  # term log(p/q)
	#print(Intercept2)
	b = Intercept1 + Intercept2
	W = numpy.log((Prob_pos/(1-Prob_pos))*((1-Prob_neg)/Prob_neg))
	#print(a.shape == Prob_pos.shape)
	return (W, b)

def Predict(W, b, X_test, Y_test):
	#print(X_test.dot(W.T))
	Y_hat = numpy.sign(X_test.dot(W.T))
	#print(Y_hat)
	return Y_hat

train_small = numpy.array(pd.read_csv("train_small.csv", header=0))
#print(train_small.shape)
X_train_small = train_small[:,0:-1]
Y_train_small = train_small[:,-1]
#print(X_train_small[1:20, -1])
#print(Y_train_small[1:20])
(W, b) = Model(X_train_small, Y_train_small)   # Training Model to obtain Weights and bais
#print(W)

test = numpy.array(pd.read_csv("test.csv", header=0))
X_test = test[:,0:-1]
Y_test = test[:,-1]
#print(X_test.shape)
p = Predict(W,b,X_test, Y_test)
print ('Train Accuracy: %f', ((Y_test[numpy.where(p == Y_test)].size / float(Y_test.size)) * 100.0))









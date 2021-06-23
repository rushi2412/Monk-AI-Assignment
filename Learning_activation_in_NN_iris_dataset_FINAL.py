# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:16:39 2021

@author: Rushi
"""
########****************please provide the path of local directory where you have saved the iris.csv dataset******************#########

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import f1_score

data = shuffle(pd.read_csv("Iris.csv")).drop(['Id'],axis=1)
dummies = pd.get_dummies(data['Species'])
data = pd.concat([(data.drop(['Species'],axis= 1)),dummies],axis = 'columns')
X = data.drop(['Iris-setosa','Iris-versicolor','Iris-virginica'],axis= 1).values 
Y = data.iloc[:,4:].values
data = data.values
X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)

X_train =X_train.T
Y_train = Y_train.T

X_test =X_test.T
Y_test = Y_test.T
def define_structure(X,Y):
    input_unit = X.shape[0]
    hidden_unit =  4
    output_unit = Y.shape[0]
    return (input_unit,hidden_unit,output_unit)
(input_unit,hidden_unit,output_unit) = define_structure(X_train,Y_train)

def parameter_initilization(input_unit,hidden_unit,output_unit):
    
    W1 = np.random.randn(hidden_unit,input_unit)*1
    B1 = np.zeros((hidden_unit,1))
    W2 = np.random.randn(output_unit,hidden_unit)*1
    B2 = np.zeros((output_unit,1))
    K0 = np.random.randint(10)*0.1
    K1 = np.random.randint(10)*0.1
    parameters = {"W1" : W1,
                  "W2" : W2,
                  "B1" : B1,
                  "B2" : B2,
                  "K0" : K0,
                  "K1" : K1}
    return parameters

def activation_function(Z,K0,K1):
    return K0 + (Z*K1)

def softmax(Z):
    e = np.exp(Z)
    return e / e.sum(axis = 0)

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    B1 = parameters['B1']
    W2 = parameters['W2']
    B2 = parameters['B2']
    K0 = parameters['K0']
    K1 = (parameters['K1'])
    
    Z1 = np.dot(W1, X) + B1
    A1 = activation_function(Z1,K0,K1)
    Z2 = np.dot(W2, A1) + B2
    A2 = softmax(Z2)
    cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    
    return A2, cache

def cross_entropy_loss(A2 , Y , parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = - np.sum(logprobs)
    return cost 


def backward_propagation(parameters, cache, X, Y):
    #number of training example
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
   
    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot( W2.T,dZ2)
    dZ1 = dZ1 = np.dot(W1 , dA1)
    dW1 = (1/m) * np.dot(dZ1, X.T) 
    dB1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    dk0 = np.average(dA1)
    dk1 = np.average(np.multiply(dA1, Z1))
    
    grads = {"dW1": dW1, "dB1": dB1, "dW2": dW2,"dB2": dB2, "dk0":dk0, "dk1":dk1}
    
    return grads

def gradient_descent(parameters, grads, learning_rate = 0.01):
    W1 = parameters['W1']
    B1 = parameters['B1']
    W2 = parameters['W2']
    B2 = parameters['B2']
    K0 = parameters['K0']
    K1 = parameters['K1']
    
   
    dW1 = grads['dW1']
    dB1 = grads['dB1']
    dW2 = grads['dW2']
    dB2 = grads['dB2']
    dk0 = grads['dk0']
    dk1 = grads['dk1']
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    K0 = K0 - learning_rate * dk0
    K1 = K1 - learning_rate * dk1
    
    parameters = {"W1": W1, "B1": B1,"W2": W2,"B2": B2, "K0":K0, "K1":K1}
    
    return parameters

def neural_network_model(X, Y, hidden_unit, num_iterations = 1500):
    np.random.seed(3)
    input_unit = define_structure(X, Y)[0]
    output_unit = define_structure(X, Y)[2]
    
    parameters = parameter_initilization(input_unit,hidden_unit,output_unit)
    
   
    W1 = parameters['W1']
    B1 = parameters['B1']
    W2 = parameters['W2']
    B2 = parameters['B2']
    K0 = parameters['K0']
    K1 = parameters['K1']
    
    K00 = []
    K11 = []
    epoch = []
    loss = []
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = cross_entropy_loss(A2 , Y , parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = gradient_descent(parameters, grads)
        K0 = parameters["K0"]
        K1 = parameters["K1"]
        
        K00.append(K0)
        K11.append(K1)
        epoch.append(i)
        loss.append(cost)
        #if i % 5 == 0:
        #    print ("Cost after iteration %i: %f" %(i, cost))
    return parameters, K00,K11,epoch , loss

parameters,K0,K1,epoch,loss = neural_network_model(X_train, Y_train, 4, num_iterations=1500)
parameters_test,K0_test,K1_test,epoch_test,loss_test = neural_network_model(X_test, Y_test, 4, num_iterations=1500)

def prediction(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    output  = np.round(A2)
    predictions = (A2.argmax(axis=0))
    
    return predictions, output

predictions,output = prediction(parameters, X_train)
pred = Y_train.argmax(axis=0)
accuracy = ((np.sum(pred == predictions))/Y_train.shape[1])*100
F1_score = f1_score(y_true=Y_train, y_pred=output, average='weighted')
print ('Accuracy Train: %f' % accuracy)
print ('F1 Score Train: %f' % F1_score)

predictions1 , output1= prediction(parameters, X_test)
pred1 = Y_test.argmax(axis=0)
accuracy1 = ((np.sum(pred1 == predictions1))/Y_test.shape[1])*100
F1_score1 = f1_score(y_true=Y_test, y_pred=output1, average='weighted')
print ('Accuracy Test: %f' % accuracy1)
print ('F1 Score Test: %f' % F1_score)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch, K0, color='lightblue', linewidth=3,label='K0')
ax.plot(epoch, K1, color='darkgreen', linewidth=3,label='K1')
ax.legend()
plt.xlabel("epoch")
plt.ylabel("Values")
plt.title("Parameter Update on epoch")

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(epoch, loss, color='lightblue', linewidth=3,label='loss Train')
ax1.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss function Vs Epoch")


    
    
    

    

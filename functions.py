#This file includes a set of functions I am using in implementing my backprop
import numpy as np

# Definitions of activation functions and their derivatives
def relu(a):
    return np.maximum(0, a)

def soft_max(a):
    my_exp = np.exp(a-np.max(a))
    return my_exp/my_exp.sum(axis=0, keepdims=True)

def relu_derivative(b):
    return (b > 0) * 1

#aka mu_n in the lecture notes
def sigmoid(a):
    return 1/(1+np.exp(-a))

def sigmoid_derivative(b):
    s = 1 / (1 + np.exp(-b))
    return s * (1 - s)

def tanh_derivative(b):
    return 1 - (np.tanh(b))**2


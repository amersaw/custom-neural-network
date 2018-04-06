from math import exp
import numpy as np

def sigmoid(v):
	return 1.0 / (1.0 + exp(-v))

def sigmoid_derivative(v):
	return v * (1.0 - v)


def tanh(v):
	return np.tanh(v)

def tanh_derivative(v):
	return 1- np.tanh(v)**2

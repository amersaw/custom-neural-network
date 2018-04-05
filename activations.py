from math import exp
def sigmoid(v):
	return 1.0 / (1.0 + exp(-v))

def sigmoid_derivative(v):
	return v * (1.0 - v)


def tanh(v):
	import numpy as np
	return np.tanh(v)

def tanh_derivative(v):
	import numpy as np
	return 1- np.tanh(v)**2 

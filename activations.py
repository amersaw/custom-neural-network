from math import exp
def sigmoid(v):
	return 1.0 / (1.0 + exp(-v))
	
# Calculate the derivative of an neuron output
def sigmoid_derivative(output):
	return output * (1.0 - output)
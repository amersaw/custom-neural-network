from random import random
from random import seed
from activations import *

seed(42)

class CustomNN:

	def __init__ (self, input_count, hidden_count, output_count, aFunc = sigmoid,aFuncDir = sigmoid_derivative):
		self.input_count = input_count
		self.hidden_count = hidden_count
		self.output_count = output_count
		self.net = list()
		self.add_layer(input_count, hidden_count)
		# print(self.net[-1])
		self.add_layer(hidden_count, output_count)
		# print(self.net[-1])
		self.activationFunction = aFunc
		self.activationFunctionDerivative = aFuncDir
		# print(self.net)

	def add_layer(self, prev_count, curr_count):
		res = list()
		for i in range(curr_count):
			item = {}
			item['w'] = [random() for j in range(prev_count)]
			item['b'] = random()
			res.append(item)
		self.net.append(res)

	def activate(self, layer, values):
		res = layer['b']
		for i in range(len(layer['w'])): # can be done usind dot product
			res += layer['w'][i] * values[i]

		return res
	def forward_propagate(self, row):
		input = row
		for layer in self.net:
			current_output = []
			for neuron in layer:
				activation = self.activate(neuron, input)
				neuron['output'] = self.activationFunction(activation)
				current_output.append(neuron['output'])
			input = current_output
		print(self.net)
		return input

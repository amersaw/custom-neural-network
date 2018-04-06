from random import random
from activations import *

class CustomNN:
	def __init__ (self, input_count, hidden_count, output_count, l_rate,
					aFunc = sigmoid, aFuncDer = sigmoid_derivative):
		self.input_count = input_count
		self.hidden_count = hidden_count
		self.output_count = output_count
		self.learning_rate = l_rate
		self.net = list()
		self.add_layer(input_count, hidden_count)
		self.add_layer(hidden_count, output_count)
		self.activationFunction = aFunc
		self.activationFunctionDerivative = aFuncDer

	def add_layer(self, prev_count, curr_count):
		res = list()
		for i in range(curr_count):
			item = {}
			item['w'] = [random() for j in range(prev_count)]
			item['b'] = random()
			res.append(item)
		self.net.append(res)

	def activate(self, neuron, values):
		res = neuron['b']
		for i in range(len(neuron['w'])): # can be done usind dot product
			res += neuron['w'][i] * values[i]
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
		# print(self.net)
		return input

	def back_propagate(self,actual):
		for i in range(len(self.net)-1, -1, -1):
			layer = self.net[i]
			for j in range(len(layer)):
				neuron = layer[j]
				if i == len(self.net) -1: # the last layer
					error = actual[j] - neuron['output']
				else:
					error = sum ([nextneuron['w'][j] * nextneuron['change'] for nextneuron in self.net[i+1]])
				neuron['change'] = error * self.activationFunctionDerivative(neuron['output']);

	def update_weights(self, record):
		for i in range(len(self.net)): # go across layers
			prev_output = record
			if i != 0:
				prev_output = [neuron['output'] for neuron in self.net[i-1]]
			for neuron in self.net[i]:
				for j in range(len(neuron['w'])):
					neuron['w'][j] += self.learning_rate * neuron['change'] * prev_output[j]
				neuron['b'] += self.learning_rate * neuron['change']
	def do_train_step (self, record, result):
		outputs = self.forward_propagate(record)
		expected = [0 for i in range(self.output_count)]
		#print(result)
		expected[int(result)] = 1
		loss = sum([(expected[j] - outputs[j])**2 for j in range(self.output_count)])
		self.back_propagate(expected)
		self.update_weights(record)
		return loss

	def train_network(self, data, epoch_count,print_loss=False):
		for epoch in range(epoch_count):
			epoch_loss = 0
			i = 0
			for row in data:
				loss = self.do_train_step(row[0:-1],row[-1])
				i += 1.0
				epoch_loss += loss
				if print_loss:
					print('loss : %.3f\t' %(epoch_loss / i),end='',flush=True)
			if epoch_loss <= 0.1:
				print('>epoch=%d, error=%.3f' % (epoch, epoch_loss))
				break
			if epoch % 1000 ==0:
				print('>epoch=%d, error=%.3f' % (epoch, epoch_loss))
	def predict(self, row):
		outputs = self.forward_propagate(row)
		return outputs.index(max(outputs))

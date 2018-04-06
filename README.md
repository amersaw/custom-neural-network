# Custom Neural Network
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)

This project is an implementation of a very simple neural network with 1 hidden
layer from scratch without using any external tools.


## Notes

The following tutorials was a great guide for me

- [Implementing a Neural Network from Scratch in Python – An Introduction](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
- [How to Implement the Backpropagation Algorithm From Scratch In Python](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

Implementing the neural network manually requires understanding the math
beneath it and how the values are travelling through the neurons within the
network and the method used to train the network.

I'm suggesting watch the following Youtube series for a quick introduction to
what are we doing in general

[Neural Networks Demystified (1 of 7)](https://www.youtube.com/watch?v=bxe2T-V8XRs)


## Examples

### XOR

First let's ask [**Why can't the XOR-problem be solved by a one-layer perceptron?**](https://www.quora.com/Why-cant-the-XOR-problem-be-solved-by-a-one-layer-perceptron/answer/Peter-Cadow-Kopciak)

The separating hyperplane of Single layer perceptron is given by:
w.x+b=0, which is linear function of input vector x.

But if you see the plot of XOR
![](https://qph.ec.quoracdn.net/main-qimg-a5d601ec5cd00795905c02f8271fc704)

The positive and negative points cannot be separated by a linear line, or effectively, there does not exist a (linear) line that can separate the positive and negative points. This is why XOR problem cannot be solved by One layer perceptron.

Using the neural network we built here we can build a network with 2 hidden neurons that will be able to solve it.

Our input data will be the XOR truth table which is simply

| X | Y | Result |
|---|:--|:------:|
| 0 | 0 |   0    |
| 0 | 1 |   1    |
| 1 | 0 |   1    |
| 1 | 1 |   0    |

In the [xor.py](xor.py) code example I'm demonstrating how the XOR problem is being solved using a neural network with 2 hidden neurons, where the `AND`, `OR` problems can simply be solved using neural network with only 1 neuron in the hidden network.

### MNIST


## License

[Apache License 2.0](LICENSE)

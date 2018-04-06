
import os
import struct
import numpy as np
from CustomNN import CustomNN
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

def read_mnist(dataset = "training", path = "data"):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])
    return (lbl,img)

def consoleImg(img):
	res = ''
	for i in img:
		for j in i:
			res += '#' if j>50 else ' '
		res += '\n'
	return res

if __name__ == '__main__':
    lbl,img = read_mnist()
    mnist_network = CustomNN(28*28, 8, 3, 0.3)
    dataset = []
    for i,l in enumerate(lbl):
        if l <3:
            t = np.append(img[i].reshape([28*28])/255, [l])
            dataset.append(t)
    mnist_network.train_network(dataset, 100,True)

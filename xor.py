from CustomNN import CustomNN
from activations import *
from random import seed

seed(1)

dataset_or  = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
dataset_and = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
dataset_xor = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]

def test(hidden_count, dataset, name):
    print('\n\n\t\t\t\t####  %s (%d neuron in hidden layer) ####' % (name, hidden_count))
    network = CustomNN(2, hidden_count, 2, 2)#,tanh, tanh_derivative)
    network.train_network(dataset, 100000)
    true_result = 0.
    count = 0.
    for i in dataset:
        prediction = network.predict(i[0:-1])
        count += 1
        true_result += 1. if prediction == i[-1] else 0.
        print('Input : %d , %d -> Actual: %d, Prediction: %d' % (i[0],i[1],i[2], prediction))
    print('\t\t\t\t ---> Accuracy:%.0f%% ' % (true_result/count*100))

test(2,dataset_xor, 'XOR')
test(1,dataset_and, 'AND')
test(1,dataset_or, 'OR')

import numpy as np
import matplotlib.pyplot as plt 
from neural_network import NeuralNetwork

np.random.seed(0)

nn = NeuralNetwork(3, 4, 2, 3)

inputs = np.array([[0.1, 0.5],[-0.5, 0.9]])
y = nn.one_hot(np.array([2,0]))

#fwd = nn.forward(inputs.T)
#print(fwd)

#nn.backpropagation(2, inputs.T, y, 2, 0.01)

#nn.train(inputs, y, 0.01, 2)
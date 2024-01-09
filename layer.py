import numpy as np

class Layer:
    output_layer: bool
    n_neurons: int

    def __init__(self, n_inputs, n_neurons, output_layer):
        self.weights = np.random.rand(n_neurons, n_inputs)
        self.biases = np.random.rand(n_neurons, 1)
        self.n_neurons = n_neurons
        self.output_layer = output_layer

    def forward(self, inputs):
        self.z = np.dot(self.weights, inputs) + self.biases
        if not self.output_layer:
            self.ReLU_activation(self.z)
        else:
            self.softmax(self.z)

    def ReLU_activation(self, inputs):
        self.a = np.maximum(0, inputs)

    def softmax(self, inputs): #axis=0 por columnas, axis=1 por filas
        self.a = (np.exp(inputs - np.max(inputs, axis=0, keepdims=True)) 
                  / np.sum(np.exp(inputs - np.max(inputs, axis=0, keepdims=True)), axis=0, keepdims=True))
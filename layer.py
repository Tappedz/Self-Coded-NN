import numpy as np

class Layer:
    output_layer: bool

    def __init__(self, n_inputs, n_neurons, output_layer):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.random.rand(1, n_neurons)
        self.output_layer = output_layer

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if not self.output_layer:
            self.output = self.ReLU_activation(self.output)
        else:
            self.output = self.softmax(self.output)

    def ReLU_activation(self, inputs):
        self.output = np.maximum(0, inputs)

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
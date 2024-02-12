import numpy as np

class Layer:
    output_layer: bool
    n_neurons: int
    dropout: float

    def __init__(self, n_inputs, n_neurons, dropout, output_layer):
        self.weights = np.random.rand(n_neurons, n_inputs).round(4)
        self.biases = np.random.rand(n_neurons, 1).round(4)
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.output_layer = output_layer
        
    def forward(self, inputs):
        self.z = (np.dot(self.weights, inputs) + self.biases).round(4)
        if not self.output_layer:
            self.ReLU_activation(self.z)
        else:
            self.softmax(self.z)

    def ReLU_activation(self, inputs):
        prob = np.random.rand()
        if prob >= self.dropout:
            self.a = np.maximum(0, inputs).round(4)
        else:
            self.a = np.zeros(inputs.shape)

    def softmax(self, inputs): #axis=0 por columnas, axis=1 por filas
        self.a = (np.exp(inputs - np.max(inputs, axis=0, keepdims=True)) 
                  / np.sum(np.exp(inputs - np.max(inputs, axis=0, keepdims=True)), axis=0, keepdims=True)).round(4)
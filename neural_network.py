import numpy as np
from layer import Layer

class NeuralNetwork:
    n_hid_layers: int
    n_neurons: int
    layers: []


    def __init__(self, n_hid_layers, n_neurons, n_inputs, n_outputs):
        self.n_hid_layers = n_hid_layers
        self.n_neurons = n_neurons

        for i in range(n_hid_layers):
            if i == 0:
                self.layers.append(Layer(n_inputs, n_neurons))
            else:
                self.layers.append(Layer(n_neurons, n_neurons))
        self.layers.append(Layer(n_neurons, n_outputs))

    def forward(self, inputs):
        for i in range(len(self.layers)):
            self.layers[i].forward(inputs)
            inputs = self.layers[i].output
        return inputs

    def calculate_loss(self, predictions, labels): # Cross Entropy Loss
        samples = len(predictions)
        y_pred_clipped = np.clip(predictions, 1e-7, 1-1e-7) # we make sure there is no 0 so no log(0) = inf

        if len(labels.shape) == 1: # scalars
            confidences = y_pred_clipped[range(samples), labels]
        elif len(labels.shape) == 2: # one-hot encoding
            confidences = np.sum(y_pred_clipped*labels, axis=1)

        return -np.log(confidences)


    def backpropagation(self, loss):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i]


def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1
    one_hot_labels = one_hot_labels.T
    return one_hot_labels

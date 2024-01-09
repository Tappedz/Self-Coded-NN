import numpy as np
from layer import Layer

class NeuralNetwork:
    n_layers: int
    n_neurons: int

    def __init__(self, n_layers, n_neurons, n_inputs, n_outputs):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.layers = []

        for i in range(n_layers - 1):
            if i == 0:
                self.layers.append(Layer(n_inputs, n_neurons, False))
            else:
                self.layers.append(Layer(n_neurons, n_neurons, False))
        self.layers.append(Layer(n_neurons, n_outputs, True))

    def forward(self, inputs): #inputs n_batches x n_inputs
        self.inputs = inputs.T
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.inputs)
            else:
                self.layers[i].forward(fwd)
            fwd = self.layers[i].a
        return fwd

    def calculate_loss(self, predictions, labels): # Cross Entropy Loss
        samples = len(predictions)
        y_pred_clipped = np.clip(predictions, 1e-7, 1-1e-7) # we make sure there is no 0 so no log(0) = inf

        if len(labels.shape) == 1: # scalars
            confidences = y_pred_clipped[range(samples), labels]
        elif len(labels.shape) == 2: # one-hot encoding
            confidences = np.sum(y_pred_clipped*labels, axis=0)

        return -np.log(confidences)
    
    def calculate_accuracy(self, predictions, labels):
        pred_indexes = np.argmax(predictions, axis=0)
        labels_indexes = np.argmax(labels, axis=0)
        return (np.count_nonzero(pred_indexes == labels_indexes) / predictions.shape[1])

    def ReLU_derivative(self, z):
        return z > 0
        
    def backpropagation(self, layer_num, y, n_batches, alpha):
        layer = self.layers[layer_num]
        if(layer_num == len(self.layers) - 1):
            dZ = layer.a - y 
            dW = 1 / n_batches * np.dot(dZ, self.layers[layer_num - 1].a.T)
            dB = 1 / n_batches * np.sum(dZ, axis=1, keepdims=True)
            self.backpropagation(layer_num - 1, dZ, n_batches, alpha)
        elif(layer_num == 0):
            dZ = np.dot(self.layers[layer_num+1].weights.T, y) * self.ReLU_derivative(layer.z) 
            dW = 1 / n_batches * np.dot(dZ, self.inputs.T)
            dB = 1 / n_batches * np.sum(dZ, axis=1, keepdims=True)
        else:       
            dZ = np.dot(self.layers[layer_num+1].weights.T, y) * self.ReLU_derivative(layer.z) 
            dW = 1 / n_batches * np.dot(dZ, self.layers[layer_num - 1].a.T)
            dB = 1 / n_batches * np.sum(dZ, axis=1, keepdims=True)
            self.backpropagation(layer_num - 1, dZ, n_batches, alpha)
        self.adjust_params(layer, dW, dB, alpha)
        
    def adjust_params(self, layer, dW, dB, alpha):
        layer.weights = layer.weights - alpha * dW
        layer.biases = layer.biases - alpha * dB


def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1
    one_hot_labels = one_hot_labels.T
    return one_hot_labels




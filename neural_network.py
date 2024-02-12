import numpy as np
from layer import Layer

class NeuralNetwork:
    n_layers: int
    n_neurons: int
    dropout: float

    def __init__(self, n_layers, n_neurons, n_inputs, n_outputs, dropout):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.layers = []

        for i in range(n_layers - 1):
            if i == 0:
                self.layers.append(Layer(n_inputs, n_neurons, dropout, False))
            else:
                self.layers.append(Layer(n_neurons, n_neurons, dropout, False))
        self.layers.append(Layer(n_neurons, n_outputs, dropout, True))

    def forward(self, inputs): #inputs n_batches x n_inputs
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(inputs)
            else:
                self.layers[i].forward(fwd)
            fwd = self.layers[i].a
        return fwd

    def calculate_loss(self, predictions, labels): # Cross Entropy Loss
        pred = np.clip(predictions, 1e-7, 1+1e-7) # we make sure there is no 0 so no log(0) = inf
        loss = np.sum(-labels * np.log(pred))
        return loss / predictions.shape[1]
    
    def calculate_accuracy(self, predictions, labels):
        pred_indexes = np.argmax(predictions, axis=0)
        labels_indexes = np.argmax(labels, axis=0)
        return (np.count_nonzero(pred_indexes == labels_indexes) / predictions.shape[1])

    def ReLU_derivative(self, z):
        return z > 0
        
    def backpropagation(self, layer_num, inputs, y, batch_size):
        layer = self.layers[layer_num]
        if(layer_num == len(self.layers) - 1):
            dZ = layer.a - y 
            dW = 1 / batch_size * np.dot(dZ, self.layers[layer_num - 1].a.T) 
            dB = 1 / batch_size * np.sum(dZ, axis=1, keepdims=True)
            self.backpropagation(layer_num - 1, inputs, dZ, batch_size)
        elif(layer_num == 0):
            dZ = np.dot(self.layers[layer_num+1].weights.T, y) * self.ReLU_derivative(layer.z) 
            dW = 1 / batch_size * np.dot(dZ, inputs.T)
            dB = 1 / batch_size * np.sum(dZ, axis=1, keepdims=True)
        else:       
            dZ = np.dot(self.layers[layer_num+1].weights.T, y) * self.ReLU_derivative(layer.z) 
            dW = 1 / batch_size * np.dot(dZ, self.layers[layer_num - 1].a.T)
            dB = 1 / batch_size * np.sum(dZ, axis=1, keepdims=True)
            self.backpropagation(layer_num - 1, inputs, dZ, batch_size)
        self.adjust_params(layer, dW, dB)
        
    def adjust_params(self, layer, dW, dB):
        layer.weights = layer.weights - self.alpha * dW.round(4)
        layer.biases = layer.biases - self.alpha * dB.round(4)

    def one_hot(self, labels):
        one_hot_labels = np.zeros((labels.size, labels.max() + 1))
        one_hot_labels[np.arange(labels.size), labels] = 1
        one_hot_labels = one_hot_labels.T
        return one_hot_labels
    
    def train(self, X, y, alpha, batch_size, epoches):
        self.alpha = alpha
        inputs = X.T
        acc = []
        loss = []
        n_batches = inputs.shape[1] // batch_size
        #inputs = np.array_split(inputs, n_batches, axis=1)
        #y = np.array_split(y, n_batches, axis=1)
        for epoch in range(epoches):
            epoch_acc = []
            epoch_loss = []
            for i in range(n_batches):
                input_batch = inputs[:,(i*batch_size):((i+1)*(batch_size))]
                y_batch = y[:,(i*batch_size):((i+1)*(batch_size))]
                pred = self.forward(input_batch)
                self.backpropagation(self.n_layers - 1, input_batch, y_batch, batch_size)
                epoch_acc.append(self.calculate_accuracy(pred, y_batch))
                epoch_loss.append(self.calculate_loss(pred, y_batch))
            acc.append(np.mean(epoch_acc))
            loss.append(np.mean(epoch_loss))
            if epoch % 5 == 0:
                print("Epoch " + str(epoch) + ", accuracy: ", acc[epoch])
        return acc, loss
    
    def test(self, X, y, batch_size):
        inputs = X.T
        y = self.one_hot(y)
        acc = []
        loss = []
        if batch_size == 1:
            inputs = np.array_split(inputs, inputs.shape[1], axis=1)
            y = np.array_split(y, y.shape[1], axis=1)
        else:
            inputs = np.array_split(inputs, inputs.shape[1] / batch_size, axis=1)
            y = np.array_split(y, batch_size, axis=1)
        for i in range(len(inputs)):
            pred = self.forward(inputs[i])
            acc.append(self.calculate_accuracy(pred, y[i]))
            loss.append(self.calculate_loss(pred, y[i]))
        return acc, loss






import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from neural_network import NeuralNetwork

data = pd.read_csv('./mnist_data/mnist_train.csv')
test = pd.read_csv('./mnist_data/mnist_test.csv')

data = np.array(data)
test = np.array(test)

np.random.shuffle(data)

X_train = data[:,1:]
Y_train = data[:,:1]

X_test = test[:,1:]
Y_test = test[:,:1]

nn = NeuralNetwork(2, 16, 784, 10, 0)

Y_train = nn.one_hot(Y_train.T)
Y_test = nn.one_hot(Y_test.T)

train_acc, train_loss = nn.train(X_train, Y_train, alpha=0.00001, batch_size=64, epoches=200)

plt.plot(train_acc)
plt.title("Accuracy")
plt.ylabel('acc')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()

plt.plot(train_loss)
plt.title("Loss")
plt.ylabel('l')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()

test_acc = nn.test(X_test, Y_test, batch_size=64)

plt.plot(test_acc)
plt.title("Accuracy")
plt.ylabel('acc')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()
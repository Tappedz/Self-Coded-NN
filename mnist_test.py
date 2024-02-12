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

nn = NeuralNetwork(2, 64, 784, 10, 0)

Y_train = nn.one_hot(Y_train.T)
#print(X_train.shape, Y_train.shape)

acc, loss = nn.train(X_train, Y_train, alpha=0.01, batch_size=64, epoches=20)
print(loss[:4])
print(loss[-4:])

plt.plot(acc)
plt.title("Accuracy")
plt.ylabel('acc')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()

plt.plot(loss)
plt.title("Loss")
plt.ylabel('l')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()

'''
y = np.array(data['label'])
y_test = np.array(test['label'])
data = np.array(data.drop("label", axis=1))
test = np.array(test.drop("label", axis=1))

nn = NeuralNetwork(2, 64, 784, 10)

y = nn.one_hot(y)

acc, loss = nn.train(data, y, 0.01, 1000, 20)

plt.plot(acc)
plt.title("Accuracy")
plt.ylabel('acc')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()


plt.plot(loss)
plt.title("Loss")
plt.ylabel('l')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()


acc, loss = nn.test(test, y_test, 100)
plt.plot(acc)
plt.title("Accuracy")
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend('train', loc='upper left')
plt.show()
'''
#data = np.array(data)

#y = np.array(np.array_split(y, 30000, axis=0))

#print(data.head(5))
#print(y.head(5))
#print(data.shape)
#print(y.shape)

'''
plt.figure(figsize=(7,7))
idx = 20

grid_data = data.iloc[idx].to_numpy().reshape(28,28)
plt.imshow(grid_data, interpolation="none", cmap="gray")
plt.show()

print(y[idx])
'''

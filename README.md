# Self-Coded-NN
__OOP__ Implementation of a neural network. The objective of this project was to investigate deeply on neural networks and have a personal "library" to solve deep learning problems. Through the development of this project I have investigated, learned and applied machine learning theory achieving a decent knowledge on the subject and developing even more interest in __AI__.

With the implementation of the neural network, there is a simple performance test on a supervised learning problem such as the MNIST dataset (digit recognition).  

## Classes
### Layer
Based on the concept of a layer inside a neural network, this class takes care of the forward function and various activation functions. It has associated weights and neurons represented as arrays.

### Neural_Network
It manages all the layers, from the input layer, through the hidden layers until the output layer. It is responsible of forwarding inputs to its layers and it is also responsible of performing backpropagation in the training phase. It has two main functions:  
- train
- test

The NN can be built with many parameters:  
- Number of layers
- Neurons per layer
- Number of inputs
- Number of outputs
- Dropout %

## Future implementations
At the moment I'm investigating and learning about Deep Q Learning because my objective is to teach a NN to play Snake, game which I have implemented. So in the near future there will be expansions in the repository in order to be capable of achieving this goal. 

In the meantime I'm performing tests with Tensorflow in the matter of Deep Q. (I will update this readme with theory I used to help me understand and code).

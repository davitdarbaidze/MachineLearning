import numpy as np
from random import random

class MLP:


    def __init__(self, num_input=3, num_hidden=[3,5], num_outputs=2):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_input] + self.num_hidden + [self.num_outputs]
        # initiate random weights

        self.weights = []
        
        for i in range(len(layers)-1):            
            w = np.random.rand(layers[i],layers[i+1])
            
            self.weights.append(w)

        #store activasions
        activasions = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activasions.append(a)
        self.activasions = activasions
        print("act")
        print(self.activasions)

        #store derivatives
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        print("der")
        print(self.derivatives)

    def forward_propagate(self, inputs):
        activasions = inputs
        self.activasions[0] = inputs
        for i,w in enumerate(self.weights):
            # calculae net inputs
            net_inputs = np.dot(activasions,w)
            #calculate activisions
            activasions = self._sigmoid(net_inputs)
            self.activasions[i+1] = activasions
            #print(self.activasions)

        return activasions

    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))



    # dE/dW_i = (y - a_[i+1]) * s'(h_[i+1])) a_i
    # s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
    # s(h_[i+1]) = a_[i+a]


    def back_propagate(self, error, verbose = False):
        for i in reversed(range(len(self.derivatives))):
            
            activasions = self.activasions[i+1]
            delta = error * self._sigmoid_derivative(activasions) # ndarray ([0.1, 0.2] --> ndarray ([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activasions[i] # ndarray ([0.1, 0.2] --> ndarray ([[0.1], [0.2]])

            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error


    def gradient_decent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))

            derivatives = self.derivatives[i]

            weights = weights + derivatives * learning_rate
            #print("Updated W{} {}".format(i, weights))
    
    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forward_propagate(input)
                #Calculate error
                error = target - output
                #backward propagation
                self.back_propagate(error, verbose=False)
                #apply gradiant decent
                self.gradient_decent(learning_rate)

                sum_error += self._mse(target,output)
            #error report
            print("Error : {} at epoch {}".format(sum_error / len(inputs), i ))


    def _mse(self,target,output):
        return np.average((target - output) ** 2)

    def _sigmoid_derivative(self, x):
        return x * (1.0- x)
    

#creating MLP






#create inputs and targets
inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] + i[1]] for i in inputs])
#train

mlp = MLP(2, [5] ,1)
mlp.train(inputs, targets, 10, 0.1)

input = np.array([0.3, 0.7])
target = np.array([1.0])


output = mlp.forward_propagate(input)
print()
print()
print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
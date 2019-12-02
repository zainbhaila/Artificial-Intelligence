"""
File: studNet.py -- Zain Bhaila, Nov 27, 2019

Project 3 - Neural Network
Neural Network for binary output. Uses two hidden 
layers of size 8.

I pledge on my honor that I have not given or received
any unauthorized assistance on this project.
Zain Bhaila
"""

import numpy as np

def activation(z):
    '''
    tanh is used as activation function
    
    Input: a vector of elements to preform tanh on
    
    Output: a vector of elements with tanh preformed on them
    '''
    return np.tanh(z)


def sigderiv(z):
    '''
    The derivative of tanh, used to calculate deltas for back prop
    '''
    return 1-activation(z)**2

class NeuralNetwork(object):     
    '''
    This Object outlines a basic neuralnetwork and its methods
    
    We have included an init method with a size parameter:
        Size: A 1D array indicating the node size of each layer
            E.G. Size = [2, 4, 1] Will instantiate weights and biases with 2 
            input nodes, 1 hidden layer with 4 nodes, output layer with 1 node
        
        test_train defines the sizes of the input and output layers
    
    In this network for simplicity all nodes in a layer are connected to all 
    nodes in the next layer, and the weights and biases and intialized as such. 
    E.G. In a [2, 4, 1] network each of the 4 nodes in the inner layer 
    will have 2 weight values and one biases value.
    '''

    def __init__(self, size, seed=42):
        '''
        Here the weights and biases specified above will be 
        instantiated to random values. 
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * 
        np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

    def calcDeltas(self, results, pre, activations, correct):
        '''
        Calculates deltas for each node in the network.
        Used in backpropagate.
        Takes in expected y, output y, preactivations, activations.
        Returns deltas.
        '''
        deltas = []
        finaldelta= np.dot(np.subtract(correct, results),
        sigderiv(list(reversed(pre))[0]))
        
        incomingweights = (self.weights*finaldelta)[0][2][0]
        presigderiv = sigderiv(list(reversed(pre))[1])
        finals = [incomingweights[i]*presigderiv[i] 
        for i in range(len(incomingweights))]
        
        incomingweights = (self.weights*finaldelta)[0][1]
        presigderiv = sigderiv(list(reversed(pre))[2])
        finals2 = [np.array([sum(incomingweights[i]*presigderiv[i])]) 
        for i in range(len(incomingweights))]
        
        deltas.append(finals2)
        deltas.append(finals)
        deltas.append([finaldelta[0]])
        #print(incomingweights, "\n\n",sigderiv(list(reversed(pre))[2]))
        #print(deltas)
        
        return deltas

    def backpropagate(self, deltas, activations):
        '''
        Backpropogation for the neural network.
        Takes in deltas and activations values for each node.
        Updates weight using formula as described in slides.
        No return value
        '''
        alpha = self.alpha
        #print(deltas, "\n\n", self.weights, "\n\n", self.biases)
        for i in range(self.size[0]): # update weights for first hidden layer
            for j in range(self.size[1]):
                self.weights[0][j][i] += activations[0][i][0]*alpha*deltas[0][j]
        for j in range(self.size[1]): # update weights for second hidden layer
            for i in range(self.size[1]):
                self.weights[1][j][i] += activations[1][i][0]*alpha*deltas[1][j]
            self.biases[0][j][0] += alpha*deltas[0][j]
        for j in range(self.size[2]): # update weights for output layer
            self.weights[2][0][j] += activations[2][j][0]*alpha*deltas[2][0]
            self.biases[1][j][0] += alpha*deltas[1][j]
        self.biases[2][0][0] += alpha*deltas[2][0]
        #print(self.weights, "\n\n", self.biases)
        
    def forward(self, input):
        '''
        Perform a feed forward computation 
        Parameters:

        input: data to be fed to the network with (shape described in spec)

        Returns:

        the output value(s) of each example as ‘a’

        The values before activation was applied after the input was 
        weighted as ‘pre_activations’

        The values after activation for all layers as ‘activations’

        You will need ‘pre_activaitons’ and ‘activations’ to update the network
        '''
        a = input
        pre_activations = [] # pre_activations are in_i
        activations = [a] # activations are a_i
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
            pre_activations.append(z)
            activations.append(a)
        #print(a,"\n",pre_activations,"\n",activations)
        return a, pre_activations, activations

    def train(self, X, y):
        '''
        Train the neural network.
        X is input values, y is correct outputs.
        Returns final output activation value.
        '''
        #print(self.weights,"\n\n",self.biases)
        for loop in range(0,10): # 10 epochs
            self.alpha = (.1 if loop < 5 else .05) # change alpha for backprop
            for i in range(len(X[0])): # batch size of 1
                x = [0 for i in range(self.size[0])]
                y2 = [[y[0][i]]]
                for j in range(self.size[0]):
                    x[j] = [X[j][i]]
                (a, pre, act) = self.forward(x)
                deltas = self.calcDeltas(a, pre, act, y2)
                self.backpropagate(deltas, act)
        #print(self.weights,"\n\n",self.biases)
        return a

    def predict(self, a):
        '''
        Input: a: list of list of input vectors to be tested

        This method will test a vector of input parameter vectors of the same 
        form as X in test_train and return the results (Zero or One) that your 
        trained network came up with for every element.

        This method does this the same way the included forward method moves an 
        input through the network but without storying the previous values 
        (which forward stores for use with the delta function you must write)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions
    
def test_train(X, y):
    '''
    This is the function that is used to test the network.

    It first instantiates the network.

    It then trains the network given the passed data, where x is the parameters:
            [[1rst parameter], [2nd parameter], [nth parameter]]
        
            Where if there are 100 training examples each of the n lists inside 
            the list above will have 100 elements
            
        Y is the target which is guaranteed to be binary:
        Y will be of the form: 
            [[1, 0, 0, ...., 1, 0, 1]]
    '''
    inputSize = np.size(X, 0)
    
    # the sizes of the input and output layer (inputSize and 1) must NOT CHANGE
    retNN = NeuralNetwork([inputSize, 8, 8, 1])
    # train network here
    retNN.train(X, y)
    
    # then the function MUST return TRAINED nueral network
    return retNN
    
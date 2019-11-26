import numpy as np

def activation(z):
    '''
    Sigmoid function on a vector this is included for use as your activation function
    
    Input: a vector of elements to preform sigmoid on
    
    Output: a vector of elements with sigmoid preformed on them
    '''
    return 1 / (1 + np.exp(-z))


def sigderiv(z):
    '''
    The derivative of Sigmoid, you will need this to preform back prop
    '''
    return activation(z) * (1 - activation(z))

class NeuralNetwork(object):     
    '''
    This Object outlines a basic neuralnetwork and the methods that it will need to function
    
    We have included an init method with a size parameter:
        Size: A 1D array indicating the node size of each layer
            E.G. Size = [2, 4, 1] Will instantiate weights and biases for a network
            with 2 input nodes, 1 hidden layer with 4 nodes, and an output layer with 1 node
        
        test_train defines the sizes of the input and output layers, but the rest is up to your implementation
    
    In this network for simplicity all nodes in a layer are connected to all nodes in the next layer, and the weights and
    biases and intialized as such. E.G. In a [2, 4, 1] network each of the 4 nodes in the inner layer will have 2 weight values
    and one biases value.
    
    '''

    def __init__(self, size, seed=42):
        '''
        Here the weights and biases specified above will be instantiated to random values
        Your network will change these values to fit a certain dataset by training
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

    def calcDeltas(self, results, pre, activations, correct):
        deltas = []
        finaldelta= np.dot(np.subtract(correct, results),sigderiv(list(reversed(pre))[0]))
        incomingweights = (self.weights*finaldelta)[0][1][0]
        presigderiv = sigderiv(list(reversed(pre))[1])
        finals = [incomingweights[i]*presigderiv[i] for i in range(len(incomingweights))]
        deltas.append(finals)
        deltas.append([finaldelta[0]])
        #print(finaldelta,"\n\n",finals,"\n\n",deltas)
        return deltas

    def backpropagate(self, deltas, activations):
        out = []
        alpha = .1
        #print(deltas, "\n\n", activations,"\n\n",self.weights, "\n\n", self.biases)
        for i in range(self.size[0]):
            self.weights[0][0][i] += activations[0][i][0]*alpha*deltas[0][0]
            self.weights[0][1][i] += activations[0][i][0]*alpha*deltas[0][1]
            self.weights[0][2][i] += activations[0][i][0]*alpha*deltas[0][2]
            self.weights[0][3][i] += activations[0][i][0]*alpha*deltas[0][3]
        self.weights[1][0][0] += activations[1][0][0]*alpha*deltas[1][0]
        self.weights[1][0][1] += activations[1][1][0]*alpha*deltas[1][0]
        self.weights[1][0][2] += activations[1][2][0]*alpha*deltas[1][0]
        self.weights[1][0][3] += activations[1][3][0]*alpha*deltas[1][0]
        self.biases[0][0][0] += alpha*deltas[0][0]
        self.biases[0][1][0] += alpha*deltas[0][1]
        self.biases[0][2][0] += alpha*deltas[0][2]
        self.biases[0][3][0] += alpha*deltas[0][3]
        self.biases[1][0][0] += alpha*deltas[1][0]
        #print(self.weights, "\n\n", self.biases)
        return out
        
    def forward(self, input):
        '''
        Perform a feed forward computation 
        Parameters:

        input: data to be fed to the network with (shape described in spec)

        returns:

        the output value(s) of each example as ‘a’

        The values before activation was applied after the input was weighted as ‘pre_activations’

        The values after activation for all layers as ‘activations’

        You will need ‘pre_activaitons’ and ‘activations’ for updating the network
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
        #print(self.weights,"\n\n",self.biases)
        for loop in range(0,3):
            for i in range(len(X[0])):
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
       
       This method will test a vector of input parameter vectors of the same form as X in test_train
       and return the results (Zero or One) that your trained network came up with for every element.
       
       This method does this the same way the included forward method moves an input through the network
       but without storying the previous values (which forward stores for use with the delta function you must write)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions
    
'''
This is the function that we will call to test your network.

It must instantiate a network, which we include.

It must then train the network given the passed data, where x is the parameters in form:
        [[1rst parameter], [2nd parameter], [nth parameter]]
    
        Where if there are 100 training examples each of the n lists inside the list above will have 100 elements
        
    Y is the target which is guarenteed to be binary, or in other words true or false:
    Y will be of the form: 
        [[1, 0, 0, ...., 1, 0, 1]]
        
        (where 1 indicates true and zero indicates false)

'''
def test_train(X, y):
    inputSize = np.size(X, 0)
    
    #feel free to change the inside (hidden) layers to best suite your implementation
    #but the sizes of the input layer and output layer (inputSize and 1) must NOT CHANGE
    retNN = NeuralNetwork([inputSize, 4, 1])
    #train your network here
    retNN.train(X, y)
    
    #then the function MUST return your TRAINED nueral network
    return retNN
    
# BackPropagation Neural Networks
# By: Nathan Lapp
# Purpose: To create an AI to determine parity of 5-bit inputs using BackPropagation Neural Networks.

import math
import string

inputWeights = []

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# The sigmoid function, tanh(x), gives better results than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)
    #return (1 / (1 + math.e-x))
# The derivative of the sigmoid function
def dsigmoid(y):
    #return y*1.0 - y**2
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no, weights):
        # self = object of class NN
		# ni = number of input nodes, nh = number of hidden nodes, and no = number of output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        # Activations for the nodes
		# ai = activation input, ah = activation hidden, ao = activation output
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # Create a 3 dimensional array for the weights
		# wi = weights input, wo = weights output
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
			
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = weights[i]
        for k in range(self.nh):
            for l in range(self.no):
                self.wo[k][l] = weights[ni+no+k]
        # Last change in weights for momentum   
		# ci = change input, co = change output
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
		# If the number of inputs parameter is not the number of inputs,
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # Input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # Hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # Output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

	# This is the function that handles the back propagation.
    def backPropagate(self, targets, N, M):
		# If the number of targets parameter is the not the number of outputs,
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # Calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # Calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # Update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # Update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # Calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

	# Train the AI given the training patterns, number of iterations, learning rate, and momentum factor
    #def train(self, patterns, iterations=300, N=0.5, M=0.1):
    def train(self, patterns, iterations=300, N=0.4, M=0.1):
        # N = learning rate
        # M = momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            # Print the Error percentage 3 times
            if i % 100 == 0:
                print('Percent Error of Training Set: %-.5f' % error, '%')

def Parity():
    # Teach the network to determine parity of a 5-bit input
	# pat1 is first 16 combinations of the set 2^5 combinations
    pat1 = [
        [[0,0,0,0,0], [0]],
        [[0,0,0,0,1], [1]],
        [[0,0,0,1,0], [1]],
        [[0,0,0,1,1], [0]],
		[[0,0,1,0,0], [1]],
		[[0,0,1,0,1], [0]],
		[[0,0,1,1,0], [0]],
		[[0,0,1,1,1], [1]],
		[[0,1,0,0,0], [1]],
		[[0,1,0,0,1], [0]],
		[[0,1,0,1,0], [0]],
		[[0,1,0,1,1], [1]],
		[[0,1,1,0,0], [0]],
		[[0,1,1,0,1], [1]],
		[[0,1,1,1,0], [1]],
		[[0,1,1,1,1], [0]]
    ]
	# pat2 is the total set of combinations of 2^5 combinations
    pat2 = [
        [[0,0,0,0,0], [0]],
        [[0,0,0,0,1], [1]],
        [[0,0,0,1,0], [1]],
        [[0,0,0,1,1], [0]],
		[[0,0,1,0,0], [1]],
		[[0,0,1,0,1], [0]],
		[[0,0,1,1,0], [0]],
		[[0,0,1,1,1], [1]],
		[[0,1,0,0,0], [1]],
		[[0,1,0,0,1], [0]],
		[[0,1,0,1,0], [0]],
		[[0,1,0,1,1], [1]],
		[[0,1,1,0,0], [0]],
		[[0,1,1,0,1], [1]],
		[[0,1,1,1,0], [1]],
		[[0,1,1,1,1], [0]],
		[[1,0,0,0,0], [1]],
		[[1,0,0,0,1], [0]],
		[[1,0,0,1,0], [0]],
		[[1,0,0,1,1], [1]],
		[[1,0,1,0,0], [0]],
		[[1,0,1,0,1], [1]],
		[[1,0,1,1,0], [1]],
		[[1,0,1,1,1], [0]],
		[[1,1,0,0,0], [0]],
		[[1,1,0,0,1], [1]],
		[[1,1,0,1,0], [1]],
		[[1,1,0,1,1], [0]],
		[[1,1,1,0,0], [1]],
		[[1,1,1,0,1], [0]],
		[[1,1,1,1,0], [0]],
		[[1,1,1,1,1], [1]]
    ]
	# Store the weights from the file 'weights_nvl002_CSC475.txt' into inputWeights
    global inputWeights
    with open('weights.txt') as inputFile:
        input = inputFile.readlines()
    for item in input:
        inputWeights += [float(item.strip())]
	
    # Create a network with 5 input nodes, 3 hidden nodes, and 1 output node
    n = NN(5, 3, 1, inputWeights)
    print ('Train: Pattern 1 | Test: Pattern 2')
    # Train it with the training set
    n.train(pat1)
    # Print the weights of input to hidden and hidden to output after it has trained
    #n.weights()
    # Test it with the testing set
    n.test(pat2)
	
	# Create a network with 5 input nodes, 3 hidden nodes, and 1 output node
    m = NN(5, 3, 1, inputWeights)
    print ('Train: Pattern 2 | Test: Pattern 2')
    # Train it with the training set
    m.train(pat2)
	# Print the weights of input to hidden and hidden to output after it has trained
    #m.weights()
    # Test it with the testing set
    m.test(pat2)

if __name__ == '__main__':
    Parity()
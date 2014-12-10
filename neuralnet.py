import sys
import math
import random
import numpy as np



class NeuralNet:

    def __init__(self, num_in, num_hidden, num_out, learn_rate):
        # number of nodes at each layer
        self.num_in = num_in+1
        self.num_hidden = num_hidden
        self.num_out = num_out
        
        # activation of nodes
        self.input_activations = np.reshape([1]*self.num_in, (1,self.num_in))
        self.hidden_activations = np.reshape([1]*self.num_hidden, (1,self.num_hidden))
        self.output_activations = np.reshape([1]*self.num_out, (1,self.num_out))
        
        # learn rate of the neuralnet
        self.learn_rate = learn_rate
        
        # weight matrices
        self.input_weights = self._create_weight_matrix(self.num_in, self.num_hidden)
        self.hidden_weights = self._create_weight_matrix(self.num_hidden, self.num_out)

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def feed(self, d):
        # append bias term
        self.input_activations = np.array(np.append(d,1)).reshape(1,self.num_in)

        # input->hidden layer
        inets = np.dot(self.input_activations,self.input_weights)
        self.hidden_activations = np.array(map(np.vectorize(self.sigmoid), inets)).reshape(1,self.num_hidden)
        
        # hidden->output layer
        hnets = np.dot(self.hidden_activations,self.hidden_weights)
        self.output_activations = np.array(map(np.vectorize(self.sigmoid), hnets)).reshape(1,self.num_out)

    
    def backpropagate(self, y):
        # calculate hidden->output errors
        oerr = self.output_activations*(1-self.output_activations)*(y-self.output_activations)
        
        # update hidden->output weight matrix
        dhweights = np.dot(np.transpose(self.hidden_activations), oerr)
        self.hidden_weights += dhweights
        # GOOD THROUGH HERE
        
        # calculate input->hidden errors
        
        # TODO
        
        # update input->hidden weight matrix
        # TODO
    
    def _create_weight_matrix(self, row, col):
        #m = (.2)*np.random.random((row, col))-.1
        m = np.zeros((row, col))+.1
        return m
    
    def print_weights(self):
        s = "input->hidden weight matrix:\n%s\n" +\
            "hidden->output weight matrix:\n%s\n"
        print s % (self.input_weights, self.hidden_weights)
        
    def print_activations(self):
        s = "input activations:\n%s " +\
            "\nhidden activations:\n%s " +\
            "\noutput activations:\n%s\n"
        print s % (self.input_activations,
                   self.hidden_activations,
                   self.output_activations)
                   
if __name__ == '__main__':
    ### TEST DATA ###
    data = [([1,1],[0,1]),
            ([1,0],[1,1]),
            ([0,1],[1,0]),
            ([0,0],[0,0])]
    ################
    
    nn = NeuralNet(2,2,2,.1)
    for i in xrange(len(data)):
        nn.feed(np.array(data[i][0]))
        #nn.print_activations()
        nn.backpropagate(np.array(data[i][1]))
    #nn.print_weights()


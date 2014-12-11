import sys
import math
import random
import numpy as np



class NeuralNet:

    def __init__(self, num_in, num_hidden, num_out, learn_rate):
        # number of nodes at each layer
        self.num_in = num_in+1 # bias node
        self.num_hidden = num_hidden
        self.num_out = num_out
        
        # activation of nodes
        self.input_activations = np.reshape([1]*self.num_in, (1,self.num_in))
        self.hidden_activations = np.reshape([1]*self.num_hidden, (1,self.num_hidden))
        self.output_activations = np.reshape([1]*self.num_out, (1,self.num_out))
        
        # errors of nodes
        self.hidden_errors = np.reshape([1]*self.num_hidden, (1,self.num_hidden))
        self.output_errors = np.reshape([1]*self.num_out, (1,self.num_out))
        
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
        inets = np.dot(self.input_activations, self.input_weights)
        self.hidden_activations = np.array(map(np.vectorize(self.sigmoid), inets)).reshape(1,self.num_hidden)
        
        # hidden->output layer
        hnets = np.dot(self.hidden_activations, self.hidden_weights)
        self.output_activations = np.array(map(np.vectorize(self.sigmoid), hnets)).reshape(1,self.num_out)

    
    def backpropagate(self, y):
        # calculate hidden->output errors
        self.output_errors = self.output_activations \
            *(1-self.output_activations) \
            *(y-self.output_activations)
        
        # calculate input->hidden errors
        self.hidden_errors = self.hidden_activations \
            *(1-self.hidden_activations) \
            *(np.transpose((np.dot(self.hidden_weights, np.transpose(self.output_errors)))))
        
        # update hidden->output weight matrix
        dhoweights = self.learn_rate*(np.dot(np.transpose(self.hidden_activations), self.output_errors))
        self.hidden_weights += dhoweights
        
        # update input->hidden weight matrix
        dihweights = self.learn_rate*(np.dot(np.transpose(self.input_activations), self.hidden_errors))
        self.input_weights += dihweights
    
    
    def _create_weight_matrix(self, row, col):
        m = (.2)*np.random.random((row, col))-.1
        #m = np.zeros((row, col))+.1
        #m = (np.arange(row*col).reshape(row,col)+1)*.1
        return m
    
    def print_weights(self):
        s = "\ninput->hidden weight matrix:\n%s\n" +\
            "\nhidden->output weight matrix:\n%s\n"
        print s % (self.input_weights, self.hidden_weights)
        
    def print_activations(self):
        s = "\ninput activations:\n%s " +\
            "\nhidden activations:\n%s " +\
            "\noutput activations:\n%s\n"
        print s % (self.input_activations,
                   self.hidden_activations,
                   self.output_activations)
    def print_errors(self):
        s = "\noutput errors:\n%s " +\
            "\nhidden errors:\n%s "
        print s % (self.output_errors,
                   self.hidden_errors)
                   
    def train(self, data, target, epochs):
        for i in xrange(epochs):
            for instance in data:
                #print instance
                self.feed(instance[0]) # X's
                self.backpropagate(instance[1]) # Y's
    
    def test(self, data, target, verbose):
        accuracy = 0
        for instance in data:
            # feed the instance
            self.feed(instance[0])
            # map the outputs to target values
            predictions = np.array(map(np.vectorize(lambda y: 1 if y >= 0.5 else 0), self.output_activations))
            # add 1 if the prediction matches the target value
            accuracy += 1 if (predictions == instance[1]).all() else 0
            # debug printing
            if verbose: print predictions, instance[1], self.output_activations
        
        # final accuracy
        print "Prediction Accuracy: ", (float(accuracy) / len(data))*100
        
        return None


if __name__ == '__main__':
    ### TEST DATA ###
    data = [(np.array([1,1]).reshape(1,2),np.array([1]).reshape(1,1)),
            (np.array([1,0]).reshape(1,2),np.array([1]).reshape(1,1)),
            (np.array([0,1]).reshape(1,2),np.array([1]).reshape(1,1)),
            (np.array([0,0]).reshape(1,2),np.array([1]).reshape(1,1))]
    ################

    nn = NeuralNet(2, 3, 1, .1)
    nn.train(data, None, 1000)
    nn.test(data, None, False)
    
    #nn.print_activations()
    #nn.print_errors()
    #nn.print_weights()


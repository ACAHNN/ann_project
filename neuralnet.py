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
        self.hidden_activations = [np.reshape([1]*i, (1,i)) for i in self.num_hidden]
        self.output_activations = np.reshape([1]*self.num_out, (1,self.num_out))
        
        # errors of nodes
        self.hidden_errors = [np.reshape([1]*i, (1,i)) for i in self.num_hidden]
        self.output_errors = np.reshape([1]*self.num_out, (1,self.num_out))
        
        # learn rate of the neuralnet
        self.learn_rate = learn_rate
        
        # weight matrices
        self.input_weights = self._create_weight_matrix(self.num_in, self.num_hidden[0]) # input->hidden layer 1
        self.hidden_weights = [self._create_weight_matrix(i, j) for i, j in zip(self.num_hidden, self.num_hidden[1:])] # hidden->hidden layers
        self.output_weights = self._create_weight_matrix(self.num_hidden[-1], self.num_out) # last hidden->output layer


    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def feed(self, d):
        # append bias term
        self.input_activations = np.array(np.append(d,1)).reshape(1,self.num_in)

        # input->hidden layer 1
        ihnets = np.dot(self.input_activations, self.input_weights)
        self.hidden_activations[0] = np.array(map(np.vectorize(self.sigmoid), ihnets)).reshape(1,self.num_hidden[0])
        
        # hidden->hidden layers
        for i in xrange(len(self.hidden_weights)):
            hhnets = np.dot(self.hidden_activations[i], self.hidden_weights[i])
            self.hidden_activations[i+1] = np.array(map(np.vectorize(self.sigmoid), hhnets)).reshape(1,self.num_hidden[i+1])
        
        # last hidden->output layer
        honets = np.dot(self.hidden_activations[-1], self.output_weights)
        self.output_activations = np.array(map(np.vectorize(self.sigmoid), honets)).reshape(1,self.num_out)

    
    def backpropagate(self, y):
        # calculate output errors
        self.output_errors = self.output_activations \
            *(1-self.output_activations) \
            *(y-self.output_activations)
        
        # calculate last hidden layer errors
        self.hidden_errors[-1] = self.hidden_activations[-1] \
            *(1-self.hidden_activations[-1]) \
            *(np.transpose((np.dot(self.output_weights, np.transpose(self.output_errors)))))
        
        # calculate n-1 hidden layer errors
        for i in range(len(self.hidden_errors)-2,-1,-1):
            self.hidden_errors[i] = self.hidden_activations[i] \
                *(1-self.hidden_activations[i]) \
                *(np.transpose((np.dot(self.hidden_weights[i], np.transpose(self.hidden_errors[i+1])))))

        # update last hidden->output weight matrix
        dhoweights = self.learn_rate*(np.dot(np.transpose(self.hidden_activations[-1]), self.output_errors))
        self.output_weights += dhoweights
        
        # update hidden->hidden weight matrix
        for i in xrange(len(self.hidden_weights)):
            dhhweights = self.learn_rate*(np.dot(np.transpose(self.hidden_activations[i]), self.hidden_errors[i+1]))
            self.hidden_weights[i] += dhhweights
        
        # update input->hidden weight matrix
        dihweights = self.learn_rate*(np.dot(np.transpose(self.input_activations), self.hidden_errors[0]))
        self.input_weights += dihweights
    
    
    def _create_weight_matrix(self, row, col):
        m = (.2)*np.random.random((row, col))-.1
        #m = np.zeros((row, col))+.1
        #m = (np.arange(row*col).reshape(row,col)+1)*.1
        return m
    
    def print_weights(self):
        # input and output weight matrices
        s = "\ninput->hidden weight matrix:\n%s\n" +\
            "%s" +\
            "\nhidden->output weight matrix:\n%s\n"
        
        # n hidden layer matrices
        tmp = ""
        for i in xrange(len(self.hidden_weights)):
            tmp += "\nhidden->hidden weight matrix:\n" +\
                   "%s\n" % self.hidden_weights[i]
        
        print s % (self.input_weights, tmp, self.output_weights)
        
    def print_activations(self):
        # input and output activations
        s = "\ninput activations:\n%s " +\
            "%s" +\
            "\noutput activations:\n%s\n"
        
        # n hidden layer activations
        tmp = ""
        for i in xrange(len(self.hidden_activations)):
            tmp += "\nhidden activations:\n%s " % self.hidden_activations[i]
        
        print s % (self.input_activations,
                   tmp,
                   self.output_activations)
    
    def print_errors(self):
        # output layer errors
        s = "\noutput errors:\n%s " +\
            "%s\n"
        
        # n hidden layers errors
        tmp = ""
        for i in xrange(len(self.hidden_errors)-1,-1,-1):
            tmp += "\nhidden errors:\n%s " % self.hidden_errors[i]
        
        print s % (self.output_errors,
                   tmp)
                   
    def train(self, data, target, epochs):
        for i in xrange(epochs):
            for instance in data:
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
    data = [(np.array([1,1]).reshape(1,2),np.array([0]).reshape(1,1)),
            (np.array([1,0]).reshape(1,2),np.array([1]).reshape(1,1)),
            (np.array([0,1]).reshape(1,2),np.array([1]).reshape(1,1)),
            (np.array([0,0]).reshape(1,2),np.array([0]).reshape(1,1))]
#    data = [(np.array([1,0]).reshape(1,2),np.array([1,1]).reshape(1,2))]
    ################

    nn = NeuralNet(2, [3,2,4,2], 1, .1)
    nn.train(data, None, 1000)
    nn.test(data, None, True)
    
    nn.print_activations()
    nn.print_errors()
    nn.print_weights()


import sys
import time
import random
import cPickle as pickle
import numpy as np

from neuralnet import NeuralNet

def create_cross_folds(data, n):
    folds = {}
    labels = {1: [], 0:[]}
    
    random.shuffle(data)

    for instance in data:
        labels[instance[1][0][0]].append(instance)
    
    for i in xrange(n):
        folds[i] = []

    for v in labels.values():
        for i, d in enumerate(v):
            folds[i%n].append(d)
    
    return folds


def cross_validation(folds, epochs, learn_rate, n):
    averages = []
    for i in xrange(10):
        test_vals = []
        for x in xrange(len(folds.keys())):
            test_index = x%n
            test_set = folds[test_index]

            train_set = []
            for k,v in folds.items():
                if k != test_index: train_set += v
        
            nn = NeuralNet(9, [i+1], 1, learn_rate)
            nn.train(train_set, None, epochs)
            test_vals.append(nn.test(test_set, None, False))

        print "average: ", sum(test_vals) / len(test_vals)
        print ""

        averages.append(sum(test_vals)/len(test_vals))        
        
    return averages
        
def cross_validation_2(folds, epochs, learn_rate, n):
    averages = []
    for i in xrange(10):
        for j in xrange(10):
            test_vals = []
            for x in xrange(len(folds.keys())):
                test_index = x%n
                test_set = folds[test_index]

                train_set = []
                for k,v in folds.items():
                    if k != test_index: train_set += v
        
                nn = NeuralNet(9, [i+1,j+1], 1, learn_rate)
                nn.train(train_set, None, epochs)
                test_vals.append(nn.test(test_set, None, False))

            print "average: ", sum(test_vals) / len(test_vals)
            print ""

            averages.append(sum(test_vals)/len(test_vals))        
        
    return averages


def wbcd_data():

    f1 = open('data/wbcd.pkl', 'rb')
    data1 = pickle.load(f1)
    f1.close()

    folds = create_cross_folds(data1, 10)
    
    averages = cross_validation_2(folds, 10, .1, 100)

    f = open('data/wbcd_results_hidden_vary_layers.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 2, epochs = 100"
    
    pickle.dump((desc,averages), f)
    f.close()

def face_data():

    f1 = open('data/face_train.pkl', 'rb')
    f2 = open('data/face_test.pkl', 'rb')
    
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)
    
    f1.close()
    f2.close()
    
    for i in xrange(1, 122, 10):
        test_vals = []
        for j in range(10):
            nn = NeuralNet(361, [i], 1, .1)
            nn.train(data1, None, 1)
            test_vals.append(nn.test(data2, None, False))
        print "average ", sum(test_vals)/len(test_vals)
        print ""


if __name__ == '__main__':

    wbcd_data()
    #face_data()
    

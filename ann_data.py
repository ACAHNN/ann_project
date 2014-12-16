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
    for i in xrange(15):
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
    for i in xrange(15):
        for j in xrange(15):
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


def cross_validation_iterative(folds, epochs, learn_rate, n, num_points):
    
    averages = []
    test_vals = []
    fold_results = {}
    
    for x in xrange(len(folds.keys())):
        fold_results[x] = {"train": [], "test": []}
        
        test_index = x%n
        test_set = folds[test_index]

        train_set = []
        for k,v in folds.items():
            if k != test_index: train_set += v
        
        nn = NeuralNet(9, [8,8], 1, learn_rate)
        
        for j in xrange(epochs):
            nn.train(train_set, None, 1)
        
            # get train and test accuracy
            train_val = nn.test(train_set, None, False)
            test_val = nn.test(test_set, None, False)
            
            # store the accuracy results
            fold_results[x]["train"].append(train_val)
            fold_results[x]["test"].append(test_val)

        print "fold complete"

    
    # compute the average for each epoch
    train_a, test_a = [], []
    for e in xrange(epochs):
        num_train, num_test = 0, 0
        for i in xrange(len(folds.keys())):
            num_train += fold_results[i]["train"][e]
            num_test += fold_results[i]["test"][e]
        train_a.append((float(num_train)/(num_points*(n-1)))*100)
        test_a.append((float(num_test)/num_points)*100)
    
    print train_a, test_a
    return train_a, test_a


def wbcd_data():

    f1 = open('data/wbcd.pkl', 'rb')
    data1 = pickle.load(f1)
    f1.close()

    folds = create_cross_folds(data1, 10)
    
    """
    averages = cross_validation(folds, 100, .1, 10)

    f = open('data/wbcd_results_hidden_vary.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 1, epochs = 100"
    
    pickle.dump((averages,desc), f)
    f.close()

    averages = cross_validation_2(folds, 100, .1, 10)

    f = open('data/wbcd_results_hidden_vary_layers.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 2, epochs = 100"
    
    pickle.dump((averages,desc), f)
    f.close()
    """
    epochs = 400
    train_a, test_a = cross_validation_iterative(folds, epochs, .1, 10, len(data1))
    data = [([i for i in xrange(epochs)],train_a,"avg train"),([i for i in xrange(epochs)],test_a,"avg test")] 
    
    f = open('data/wbcd_results_iterative.pkl', 'wb')
    desc = "breast cancers iterative accuracy from 1 to 10000 epochs on train and test"
    
    pickle.dump((data,desc), f)
    f.close()


def face_data():

    f1 = open('data/face_train.pkl', 'rb')
    f2 = open('data/face_test.pkl', 'rb')
    
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)
    
    f1.close()
    f2.close()
    
    averages = []
    for i in xrange(1, 122, 10):
        test_vals = []
        for j in range(10):
            nn = NeuralNet(361, [i], 1, .1)
            nn.train(data1, None, 100)
            test_vals.append(nn.test(data2, None, False))
        averages.append(sum(test_vals)/len(test_vals))
        print "average ", sum(test_vals)/len(test_vals)
        print ""

    f = open('data/face_results_hidden_vary.pkl', 'wb')
    desc = "face detection averages over 10 runs on train set at each hidden " +\
           "units from 1 to 121, hidden layer = 1, epochs = 100"
    
    pickle.dump((desc,averages), f)
    f.close()
    

    averages = []
    for i in xrange(1, 122, 10):
        for x in xrange(1, 122, 10):
            test_vals = []
            for j in range(10):
                nn = NeuralNet(361, [i,x], 1, .1)
                nn.train(data1, None, 100)
                test_vals.append(nn.test(data2, None, False))
            averages.append(sum(test_vals)/len(test_vals))
            print "average ", sum(test_vals)/len(test_vals)
            print ""

    f = open('data/face_results_hidden_vary_layers.pkl', 'wb')
    desc = "face detection averages over 10 runs on train set at each hidden " +\
           "units from 1 to 121, hidden layer = 2, epochs = 100"
    
    pickle.dump((desc,averages), f)
    f.close()


if __name__ == '__main__':

    wbcd_data()
    #face_data()
    

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
    timings = []
    start_t = time.time()
    
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

        timings.append(time.time()-start_t)
        averages.append(sum(test_vals)/len(test_vals))        
        
    return averages, timings
        
def cross_validation_2(folds, epochs, learn_rate, n):
    averages = []
    timings = []
    start_t = time.time()
    for i in xrange(10):
        averages.append([])
        timings.append([])
        for j in xrange(10):
            test_vals = []
            for x in xrange(len(folds.keys())):
                test_index = x%n
                test_set = folds[test_index]

                train_set = []
                for k,v in folds.items():
                    if k != test_index: train_set += v
        
                nn = NeuralNet(9, [j+1,i+1], 1, learn_rate)
                nn.train(train_set, None, epochs)
                test_vals.append(nn.test(test_set, None, False))

            print "average: ", sum(test_vals) / len(test_vals)
            print ""

            timings[i].append(time.time()-start_t)
            averages[i].append(sum(test_vals)/len(test_vals))        

    return averages, timings


def cross_validation_iterative(folds, epochs, learn_rate, n, num_points):
    
    averages = []
    test_vals = []
    fold_results = {}
    timings = [0]*epochs

    for x in xrange(len(folds.keys())):
        fold_results[x] = {"train": [], "test": []}
        
        test_index = x%n
        test_set = folds[test_index]

        train_set = []
        for k,v in folds.items():
            if k != test_index: train_set += v
        
        nn = NeuralNet(9, [13,14], 1, learn_rate)
        
        start_t = time.time()
        for j in xrange(epochs):
            nn.train(train_set, None, 1)
        
            # get train and test accuracy
            train_val = nn.test(train_set, None, False)
            test_val = nn.test(test_set, None, False)
            
            # store the accuracy results
            fold_results[x]["train"].append(train_val)
            fold_results[x]["test"].append(test_val)
            timings[j] += time.time()-start_t
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
    
    for e in xrange(epochs):
        timings[e] = float(timings[e])/len(folds.keys())
    
    print train_a, test_a, timings
    return train_a, test_a, timings

def wbcd_data():

    f1 = open('data/wbcd.pkl', 'rb')
    data1 = pickle.load(f1)
    f1.close()

    """
    fpr, tpr = create_roc_data(data1)
    data = [(fpr, tpr, "tpr/fpr")]
    f = open('roc.pkl','wb')
    pickle.dump((data,""),f)
    f.close()
    """

    folds = create_cross_folds(data1, 10)
    epochs = 10
    
    averages,timings = cross_validation(folds, epochs, .1, 10)
    averages1,timings1 = cross_validation_2(folds, epochs, .1, 10)    
    
    data = [(timings,averages,"0 HL2 units")]
    for i in xrange(10):
        data.append((timings1[i],averages1[i], "%d HL2 units"%(i+1)))
    
    f = open('data/wbcd_results_timing.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 2, epochs = 10 (with timings)"

    pickle.dump((data,desc), f)
    f.close()
    
    """
    averages = cross_validation(folds, epochs, .1, 10)

    f = open('data/wbcd_results_hidden_vary1.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 1, epochs = 100"
    
    pickle.dump((averages,desc), f)
    f.close()

    averages = cross_validation_2(folds, epochs, .1, 10)

    f = open('data/wbcd_results_hidden_vary_layers1.pkl', 'wb')
    desc = "breast cancers averages over 10 fold cross validation varying hidden " +\
           "units from 1 to 10, hidden layer = 2, epochs = 100"
    
    pickle.dump((averages,desc), f)
    f.close()


    train_a, test_a, timings = cross_validation_iterative(folds, epochs, .1, 10, len(data1))
    data = [(timings,train_a,"avg train/compute time"),
            (timings,test_a,"avg test/compute time")] 
    
    f = open('data/wbcd_results_iterative_timings.pkl', 'wb')
    desc = "breast cancers iterative accuracy from 1 to 400 epochs on train and test with timings"
    
    pickle.dump((data,desc), f)
    f.close()
    """
    return None

def create_roc_data(data):
    
    epochs = 60
    nn = NeuralNet(9, [13,14], 1, .1)
    nn.train(data, None, epochs)
    ret = nn.test(data, None, False)

    results = []
    for row in ret:
        results.append((row[0][0][0],row[1][0][0],row[2][0][0]))

    print results[0]

    num_pos = len(filter(lambda x: x[1] == 1, results))
    num_neg = len(results)-num_pos

    results.sort(key=lambda x: x[-1])
    results.reverse()

    tp = 0
    fp = 0
    last_tp = 0

    roc_set = [[x[-2],x[-1]] for x in results]
    fpr_set = []
    tpr_set = []

    for i in range(1,len(roc_set)):
        if roc_set[i][1] != roc_set[i-1][1] and roc_set[i][0] != 1 and tp > last_tp:
            fpr = fp / float(num_neg)
            tpr = tp / float(num_pos)
            
            fpr_set.append(fpr)
            tpr_set.append(tpr)

            last_tp = tp
        if roc_set[i][0] == 1:
            tp += 1
        else:
            fp += 1

    fpr = fp / float(num_neg)
    tpr = tp / float(num_pos)

    fpr_set.append(fpr)
    tpr_set.append(tpr)

    return fpr_set, tpr_set

def ensemble_network():

    
    return None


if __name__ == '__main__':

    wbcd_data()
    

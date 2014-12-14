import sys
import time
import cPickle as pickle
import numpy as np

def scrub_wbc_dataset():
    instances = []
    with open('breast-cancer-wisconsin.data', 'rb') as fin:
        for line in fin:
            instance = line.strip().split(',')[1:]
            if '?' not in instance:
                # convert to numeric data
                instance = [int(i) for i in instance]

                # separate the feature vector and classification
                classification = [1] if instance[-1] == 4 else [0]
                instance = instance[:-1]
                instances.append((np.array(instance).reshape(1,len(instance)),
                                  np.array(classification).reshape(1,len(classification))))


    fout = open('wbcd.pkl', 'wb')
    pickle.dump(instances, fout)
    fout.close()


def scrub_face_dataset():
    instances = []
    #with open('svm.train.normgrey', 'rb') as fin:
    with open('svm.test.normgrey', 'rb') as fin:
        for line in fin:
            # scrub the instance
            instance = line.strip().split()
            instance = [float(i) for i in instance]
            
            # separate feature vector and classification
            classification = [1] if instance[-1] == 1 else [0]
            instance = instance[:-1]

            instances.append((np.array(instance).reshape(1,len(instance)),
                              np.array(classification).reshape(1,len(classification))))


    #fout = open('face_train.pkl', 'wb')
    fout = open('face_test.pkl', 'wb')
    pickle.dump(instances, fout)
    fout.close()


def scrub_letter_dataset():
    instances = []
    
    with open('letter-recognition.data', 'rb') as fin:
        for line in fin:
            # separate the feature vector and classification
            instance = line.strip().split(',')
            classification, instance = instance[0], instance[1:]
            
            # scrub the featur vector and classification
            classification = [ord(classification.lower())-97]
            instance = [int(i) for i in instance]
            
            print instance, classification
            instances.append((np.array(instance).reshape(1,len(instance)),
                              np.array(classification).reshape(1,len(classification))))


    fout = open('letters.pkl', 'wb')
    pickle.dump(instances, fout)
    fout.close()


if __name__ == '__main__':


    #scrub_wbc_dataset()
    #scrub_face_dataset()
    #scrub_letter_dataset()

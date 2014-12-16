import matplotlib.pyplot as plt
from matplotlib import cm
import cPickle as pickle
import numpy as np
if __name__ == "__main__":
    pts = [([1, 2, 3], [10, 20, 30], "hiyo!1"),
           ([1, 2, 3], [10, 20, 30], "hiyo!2"),
           ([1, 2, 3], [10, 20, 30], "hiyo!3")]
    desc = "dummy"
    
    pts, desc = pickle.load(open("data/wbcd_results_iterative.pkl"))
    #pts, desc = pickle.load(open("hw2q6.pkl"))

    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111, axisbg = 'black')
    
    ax.set_xlabel("Epoch (e)")
    ax.set_ylabel("Accuracy")
    
    #ax.set_xlabel("False positive rate")
    #ax.set_ylabel("True positive rate")

    ax.set_title("Accuracy after e Epochs")
    #ax.set_title("ROC Curve of Sonar Data Set (Mine == pos)")
    
    color_list = cm.hsv(np.linspace(0.1, .9, len(pts)))
    for (ns, ys, label), c in zip(pts, color_list):
        ax.plot(ns, ys, label = label, c = c)
    
    legend = ax.legend(loc = "lower right")
    frame = legend.get_frame()
    frame.set_facecolor('black')
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    ax.grid(True, color = 'grey')
    
    plt.savefig("writeup/figs/wbcd_iterative.pdf")
    #plt.savefig("hw2q6.pdf")
    
    #plt.show()



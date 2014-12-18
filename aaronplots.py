import matplotlib.pyplot as plt
from matplotlib import cm
import cPickle as pickle
import numpy as np
if __name__ == "__main__":
    pts = [([1, 2, 3], [10, 20, 30], "hiyo!1"),
           ([1, 2, 3], [10, 20, 30], "hiyo!2"),
           ([1, 2, 3], [10, 20, 30], "hiyo!3")]
    desc = "dummy"
    
    #pts, desc = pickle.load(open("data/roc.pkl"))
    #pts, desc = pickle.load(open("data/wbcd_results_iterative_timings.pkl"))
    pts, desc = pickle.load(open("data/wbcd_results_timing.pkl"))
    #pts, desc = pickle.load(open("hw2q6.pkl"))

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111, axisbg = 'black')
    
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Accuracy")
    #ax.set_xscale("log")
    
    #ax.set_xlabel("False positive rate")
    #ax.set_ylabel("True positive rate")

    #ax.set_title("WBCD ROC Curve")
    ax.set_title("Hidden Unit/Layer Combinations\nvs. Computational Cost")
    #ax.set_title("ROC Curve of Sonar Data Set (Mine == pos)")
    
    #ax2 = ax.twinx()
    #ax2.set_ylabel("Epochs")
    
    
    color_list = cm.hsv(np.linspace(0.1, .9, len(pts)))
    for (ns, ys, label), c in zip(pts, color_list):
        ax.plot([(float(i))/60 for i in ns], ys, label = label, c = c)
        #ax2.plot([(float(i)*10)/60 for i in ns], [j for j in xrange(400)], label = label, c = c)
        #ax.plot(ns, ys, label = label, c = c)
    
    legend = ax.legend(loc = "lower right")
    frame = legend.get_frame()
    frame.set_facecolor('black')
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    ax.grid(True, color = 'grey')
    #ax2.grid(True, color = 'grey')
    
    plt.savefig("writeup/figs/wbcd_timing.pdf")
    #plt.savefig("writeup/figs/wbcd_timing_iterative.pdf")
    #plt.savefig("writeup/figs/wbcd_roc.pdf")
    #plt.savefig("hw2q6.pdf")
    
    #plt.show()



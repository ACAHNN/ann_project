import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cPickle as pickle
import numpy as np

if __name__ == "__main__":
    desc, data = pickle.load(open("data/wbcd_results_full_matrix.pkl"))
    print desc
    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel("Number of Units in the 1st Hidden Layer")
    ax.set_ylabel("Numer of Units in the 2nd Hidden Layer")
    
    ax.set_yticks(range(0,11))
    ax.set_xticks(range(0,10))
    ax.set_xticklabels(range(1,11))
    
    
    cmap = plt.get_cmap("jet")
    
    c = ax.imshow(data, interpolation = "none", cmap = cmap, aspect = 0.4, origin='bottom')
    #cmap = c.get_cmap()
    print type(c.to_rgba(90))

    #plt.colorbar(c)
    

    min_val, max_val = 0, data.shape[0] 
    ind_array1 = np.arange(min_val, max_val + 0.0, 1.0)
    min_val, max_val = 0, data.shape[1] 
    ind_array2 = np.arange(min_val, max_val + 0.0, 1.0)
    
    
    x, y = np.meshgrid(ind_array1, ind_array2)
    print x.shape, y.shape, data.shape
    rng = np.max(data) - np.min(data)    

    
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        #c = 'x' if i%2 else 'o' 
        val = data[x_val, y_val]
        label = "%.2f" % val
        cval = -((val - np.min(data)) / 3.0) + np.max(data)
        text = ax.text(y_val, x_val, label, va='center', ha='center', color = 'white')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])
    
    
    plt.savefig("writeup/figs/wbcd_table.pdf")

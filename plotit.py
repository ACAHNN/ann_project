import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from utils import *


if __name__ == "__main__":
    desc, data = qload("wbcd_results_full_matrix.pkl")
    print desc
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("jet")
    
    c = ax.imshow(data, interpolation = "none", cmap = cmap, aspect = 0.6)
    #cmap = c.get_cmap()
    print type(c.to_rgba(90))

    #plt.colorbar(c)
    

    min_val, max_val = 0, data.shape[0] 
    ind_array1 = np.arange(min_val, max_val + 0.0, 1.0)
    min_val, max_val = 0, data.shape[1] 
    ind_array2 = np.arange(min_val, max_val + 0.0, 1.0)
    
    x, y = np.meshgrid(ind_array2, ind_array1)
    print x.shape, y.shape, data.shape
    rng = np.max(data) - np.min(data)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        #c = 'x' if i%2 else 'o' 
        val = data[y_val, x_val]
        label = "%.2f" % val
        cval = -((val - np.min(data)) / 3.0) + np.max(data)
        text = ax.text(x_val, y_val, label, va='center', ha='center', color = 'white')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])
    plt.savefig("out.pdf")

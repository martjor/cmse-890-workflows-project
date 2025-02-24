from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt 
import numpy as np 

def grad_plot(x, y, c, cmap):
    colors = cmap(c[:-1])

    # Create segments
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ax = plt.gca()
    lc = LineCollection(segments, colors=colors, linewidth=1.0,zorder=1)
    ax.add_collection(
        lc,
    )

    return lc
import numpy as np
import matplotlib.pyplot as plt


def cyclical_plot(num_points, ymax, ymin, triangle_width):
    x = []
    y = []
    stepsize = triangle_width * 1.0 / 2
    for it in range(num_points):
        x.append(it)
        cycle = np.floor(1 + it / (2 * stepsize))
        xt = np.abs(it / stepsize - 2 * (cycle) + 1)
        yt = ymin + (ymax - ymin) * (1 - xt)
        y.append(yt)
    plt.plot(x, y)
    plt.show()

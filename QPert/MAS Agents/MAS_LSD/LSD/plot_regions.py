import matplotlib.pyplot as plt
import pylab
import matplotlib
from Mas_PoinNet.MAS.Helper import Load_H5_Dataset
import time
from MAS_LSD.LSD.lsd import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cm = plt.get_cmap("RdYlGn")

# get data set
X, y = Load_H5_Dataset()
c = X[1]

# get points labales

t0 = time.time()
d, l = Lsd(c, 2048)
print("Point to shape function took : ", time.time() - t0)


# plot points with deferent colors

def plot_regions(shape, classes):
    # color map
    # NUM_COLORS = 40
    # cm = pylab.get_cmap('gist_rainbow')
    # cgen = (cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS))
    color = [int(l%40) for l in classes]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], c=color)
    ax.set_axis_off()
    plt.show()


plot_regions(c, l)

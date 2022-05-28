import numpy as np
from matplotlib import pyplot as plt


def make_pict_unfiltered(array):
    plt.title("Unfiltered Graphic")
    plt.xlabel("w")
    plt.ylabel("H(w)")
    x_label = np.array(list(range(-array.shape[1] // 2, array.shape[1] // 2))).reshape(1, -1)
    plt.scatter(x_label, array, s=0.2)
    plt.plot(x_label, array)
    plt.show()


def make_pict_filtered(array):
    plt.title("Filtered Graphic")
    plt.xlabel("w")
    plt.ylabel("H(w)")
    x_label = np.array(list(range(-array.shape[1] // 2, array.shape[1] // 2))).reshape(1, -1)
    plt.scatter(x_label, array, s=0.2)
    plt.plot(x_label, array)
    plt.show()


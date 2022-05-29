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


    def result_visualisation(array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Pixel Values on Picture')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Pixel Value')
    ax.view_init(30, 30)
    ax.scatter(list(range(0, array.shape[0])), list(range(0, array.shape[1])), array, s=0.2)
    

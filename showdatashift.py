"""Visualizes the three data sets with bias and domain shift, using the datagen module
Visualization code mostly taken from https://adapt-python.github.io/adapt/examples/Two_moons.html"""

import numpy as np
from matplotlib import pyplot as plt

from datagen.datagen import make_moons_triplet

if __name__ == "__main__":
    Xs, ys, Xg, yg, Xt, yt = make_moons_triplet()

    x_min, y_min = np.min([Xs.min(0), Xt.min(0)], 0)
    x_max, y_max = np.max([Xs.max(0), Xt.max(0)], 0)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    ax1.set_title("Input space")
    ax1.scatter(Xg[:, 0], Xg[:, 1], label="global", edgecolors='k', c="green")
    ax1.scatter(Xs[:, 0], Xs[:, 1], label="source", edgecolors='k', c="blue")
    ax1.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="orange")
    ax1.legend(loc="lower right")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')
    plt.show()

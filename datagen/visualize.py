"""Visualization functions for datasets created with the datagen module"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def visualizeShift2D(Xs, Xg, Xt):
    """Plot the positions of the source, global and target sets, without labels"""

    _, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    ax1.set_title("Input space")
    ax1.scatter(Xg[:, 0], Xg[:, 1], label="global", edgecolors='k', c="green")
    ax1.scatter(Xs[:, 0], Xs[:, 1], label="source", edgecolors='k', c="blue")
    ax1.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="orange")
    ax1.legend(loc="lower right")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')
    plt.show()


def visualizeDecisionBoundary2D(Xs, Xt, ys, yt, model, name=None):
    """Given a trained model, plots the: labeled source data, the decision boundary,
     the position of the target data, and computes the accuracy.
     Optionally add a name to the plot title."""
    yt_pred = model.predict(Xt)
    acc = accuracy_score(yt, yt_pred > 0.5)

    x_min, y_min = np.min([Xs.min(0), Xt.min(0)], 0)
    x_max, y_max = np.max([Xs.max(0), Xt.max(0)], 0)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)
    yp_grid = model.predict(X_grid).reshape(100, 100)

    X_pca = np.concatenate((model.encoder_.predict(Xs),
                            model.encoder_.predict(Xt)))
    X_pca = PCA(2).fit_transform(X_pca)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title("Input space")
    ax1.contourf(x_grid, y_grid, yp_grid, cmap=cm.RdBu, alpha=0.6)
    ax1.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1],
                label="source", edgecolors='k', c="red")
    ax1.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1],
                label="source", edgecolors='k', c="blue")
    ax1.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="black")
    ax1.legend()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')

    ax2.set_title("PCA encoded space")
    ax2.scatter(X_pca[:len(Xs), 0][ys == 0], X_pca[:len(Xs), 1][ys == 0],
                label="source", edgecolors='k', c="red")
    ax2.scatter(X_pca[:len(Xs), 0][ys == 1], X_pca[:len(Xs), 1][ys == 1],
                label="source", edgecolors='k', c="blue")
    ax2.scatter(X_pca[len(Xs):, 0], X_pca[len(Xs):, 1],
                label="target", edgecolors='k', c="black")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.tick_params(direction='in')

    title = f"Target Acc : {acc:.3f}"
    if name:
        title = f"{name} - {title}"
    fig.suptitle(title)
    plt.show()

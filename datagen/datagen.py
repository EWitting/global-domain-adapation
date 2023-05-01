"""This module contains functions to generate datasets that simulate sample selection bias"""

import numpy as np
from sklearn.datasets import make_moons


def make_moons_triplet(
        n_source=100, n_global=100, n_target=100,
        m_source=300, m_target=300,
        bias=1,
        noise=0.05, random_state=0):
    """Generate source, global and target set, from a two moons pattern, of size n
    The source and target set are sampled with bias from the same global
    distribution, of size m. Sampling bias probabilities follow a sigmoid on one axis.
    Returns (source data, source labels, global data, global labels, target data, target labels)"""

    Gx, Gy = make_moons(n_samples=n_global,
                        noise=noise,
                        random_state=random_state)

    # First create candidate data points, with the same random seed as used for the global ones
    # Then sample from those with a bias for the source and target
    Cx, Cy = make_moons(n_samples=m_source,
                        noise=noise,
                        random_state=random_state)
    Sx, Sy = sigm_bias_sampling(Cx, Cy, n_source, bias)

    Cx, Cy = make_moons(n_samples=m_target,
                        noise=noise,
                        random_state=random_state)
    Tx, Ty = sigm_bias_sampling(Cx, Cy, n_target, -bias)

    return Sx, Sy, Gx, Gy, Tx, Ty


def sigm_bias_sampling(X: np.ndarray, Y: np.ndarray,
                       n_samples: int, bias: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample with a bias, with replacement
    Samples with a probability based on the sigmoid of the first dimension,
    scaled by the bias amount after normalization"""

    # pre-compute probabilities based on bias variable V
    V = X[:, 0]
    norm = (V - np.mean(V)) / np.std(V)
    prob = 1 / (1 + np.exp(-norm*bias))

    # sample randomly and select based on bias probability until arrays are filled
    n = 0
    X_ = np.empty((n_samples, X.shape[1]))
    Y_ = np.empty((n_samples, ))
    while n < n_samples:
        i = np.random.randint(0, X.shape[0])
        if np.random.random() < prob[i]:
            X_[n] = X[i]
            Y_[n] = Y[i]
            n += 1
    return X_, Y_

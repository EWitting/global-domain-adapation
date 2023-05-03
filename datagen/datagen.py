"""This module contains functions to generate datasets that simulate sample selection bias"""

import numpy as np
from sklearn.datasets import make_moons


def make_moons_triplet(
        n_source=100, n_global=100, n_target=100,
        m_source=300, m_target=300,
        bias=1, force_balance=False,
        noise=0.05, random_state=0):
    """Generate source, global and target set, from a two moons pattern, of size n
    The source and target set simulate covariance shift, from sample selection bias
    distribution, of size m. Sampling bias probabilities follow a sigmoid on one axis.
    If force_balance is true, the source and target will have the same class balance as global,
      but the conditional class probability might change
    Returns (source data, source labels, global data, global labels, target data, target labels)"""

    Gx, Gy = make_moons(n_samples=n_global,
                        noise=noise,
                        random_state=random_state)
    global_ratio = np.mean(Gy)

    # Creates candidate data points, with the same random distribution seed as the global
    Cx, Cy = make_moons(n_samples=m_source,
                        noise=noise,
                        random_state=random_state)
    if force_balance:
        # Split candidates according to labels, sample fixed amount from both, based on global ratio
        Cx0, Cy0 = Cx[Cy==0], Cy[Cy==0]
        Cx1, Cy1 = Cx[Cy==1], Cy[Cy==1]
        n1 = int(np.floor(global_ratio*n_source))
        n0 = n_source-n1
        Sx0, Sy0 = sigm_bias_sampling(Cx0, Cy0, n0, bias)
        Sx1, Sy1 = sigm_bias_sampling(Cx1, Cy1, n1, bias)

        # Combine them and shuffle
        Sx = np.concatenate((Sx0, Sx1))
        Sy = np.concatenate((Sy0, Sy1))
        p = np.random.permutation(n_source)
        Sx = Sx[p]
        Sy = Sy[p]
    else:
        Sx, Sy = sigm_bias_sampling(Cx, Cy, n_source, bias)

    Cx, Cy = make_moons(n_samples=m_target,
                        noise=noise,
                        random_state=random_state)
    if force_balance:
        # Split candidates according to labels, sample fixed amount from both, based on global ratio
        Cx0, Cy0 = Cx[Cy==0], Cy[Cy==0]
        Cx1, Cy1 = Cx[Cy==1], Cy[Cy==1]
        n1 = int(np.floor(global_ratio*n_target))
        n0 = n_target-n1
        Tx0, Ty0 = sigm_bias_sampling(Cx0, Cy0, n0, -bias)
        Tx1, Ty1 = sigm_bias_sampling(Cx1, Cy1, n1, -bias)

        # Combine them and shuffle
        Tx = np.concatenate((Tx0, Tx1))
        Ty = np.concatenate((Ty0, Ty1))
        p = np.random.permutation(n_target)
        Tx = Tx[p]
        Ty = Ty[p]
    else:
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

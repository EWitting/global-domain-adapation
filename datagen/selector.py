"""This module contains the Selector class that is used to simulate sample selection bias.
Intended use is passing it to the builder, instead of calling the methods directly."""

import numpy as np
import scipy.stats


def sample_biased(x_ : np.ndarray, y_ :  np.ndarray, n : int,
                   mean : np.ndarray, std : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use a zero-covariance gaussian distribution to sample from candidate data with replacement."""
    distr = scipy.stats.norm(loc=mean, scale=std)
    probs = distr.pdf(x_).prod(axis=1)
    probs /= probs.sum()
    sample = np.random.choice(np.arange(len(x_)), n, p=probs)
    return x_[sample], y_[sample]


class Selector:

    def __init__(self, n_global: int, n_source: int, n_target: int,
                 source_scale: float, target_scale: float, bias_dist: float):
        self.n_global = n_global
        self.n_source = n_source
        self.n_target = n_target
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.bias_dist = bias_dist

    def select(self, x: np.ndarray, y: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # split into source candidates, global, target candidates
        xs_, ys_, xg, yg, xt_, yt_ = self._make_split(x, y)

        stds = x.std(0)
        center = x.mean(0)
        bias_dir = np.random.rand(x.shape[1])
        bias = self.bias_dist * stds * bias_dir / np.linalg.norm(bias_dir)

        source_center = center + 0.5*bias
        target_center = center - 0.5*bias
        source_std = stds*self.source_scale
        target_std = stds*self.target_scale

        xs, ys = sample_biased(xs_, ys_, self.n_source, source_center, source_std)
        xt, yt = sample_biased(xt_, yt_, self.n_target, target_center, target_std)
        return xs, ys, xg, yg, xt, yt

    def _make_split(self, x: np.ndarray, y: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition the given dataset into unbiased and disjoint global, source and target.
        Permutes the data randomly first. Returns a tuple of (xs, ys, xg, yg, xt, yt).
        The global partition has size n_global, the remainder is divided based on the ratio of n_source, n_target."""
        n_original = len(x)
        assert len(x) == len(y)
        assert n_original > self.n_global
        p = np.random.permutation(n_original)
        x, y = x[p], y[p]

        # select global without bias
        xg, yg = x[:self.n_global], y[:self.n_global]

        # divide the remainder over source and target proportionally, initially without bias
        st_ratio = self.n_source / (self.n_source + self.n_target)
        m_source = int(st_ratio * (n_original - self.n_global))

        xs, ys = x[self.n_global:self.n_global + m_source], y[self.n_global:self.n_global + m_source]
        xt, yt = x[self.n_global + m_source:], y[self.n_global + m_source:]

        return xs, ys, xg, yg, xt, yt

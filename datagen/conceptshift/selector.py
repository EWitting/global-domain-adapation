import numpy as np


def _select_biased(x, y, domain, n, target_domains):
    """Select from target domains only, select with replacement if necessary to reach n."""
    idx = [i for i in range(n) if domain[i] in target_domains]
    if len(idx) > n:
        idx = idx[:n]
    else:
        rands = np.random.choice(idx, n - len(idx), replace=True)
        idx = np.concatenate((idx, rands))
    x_, y_ = x[idx], y[idx]
    return x_, y_


class DomainSelector:

    def __init__(self, n_global: int, n_source: int, n_target: int,
                 n_domains_source: int, n_domains_target: int):
        """Create a selector for simulating sample biased selection, when data originates from distinct domains.
        The global set contains every domain, the source and targets are selected from specific domains.
        The source and target domains will be different.
        :param n_global: number of unbiased samples
        :param n_source: number of biased source samples
        :param n_target: number of biased target samples
        :param n_domains_source: number of source domains selected from
        :param n_domains_target: number of target domains selected from
        """
        self.n_global = n_global
        self.n_source = n_source
        self.n_target = n_target
        self.n_domains_source = n_domains_source
        self.n_domains_target = n_domains_target

    def select(self, x: np.ndarray, y: np.ndarray, domain: np.ndarray) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select global, source, target using initialisation parameters.
        Requires n_domains_source + n_domains_target <= #domains, and #data >= self.n_global + 2
        If enough data is in the randomly selected domains, it will be sampled without replacement.
        :param x: Features as floats in shape (N,D)
        :param y: Labels in shape (N,)
        :param domain: Domain labels in shape (N,)
        :return: tuple (xs, ys, xg, yg, xt, yt), where s=source, g=global, t=target.
        """
        domains = np.unique(domain)
        assert self.n_domains_source + self.n_domains_target <= len(domains)
        assert x.shape[0] >= self.n_global + 2

        source_domains = np.random.choice(domains, self.n_domains_source, replace=False)
        remaining_domains = [d for d in domains if d not in source_domains]
        target_domains = np.random.choice(remaining_domains, self.n_domains_target, replace=False)

        # select first partition for global samples
        xg, yg = x[:self.n_global], y[:self.n_global]

        # select source and target from remaining data
        x_, y_, domain = x[self.n_global:], y[self.n_global:], domain[self.n_global:]
        xs, ys = _select_biased(x_, y_, domain, self.n_source, source_domains)
        xt, yt = _select_biased(x_, y_, domain, self.n_target, target_domains)

        return xs, ys, xg, yg, xt, yt

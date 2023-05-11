"""Contains builder class for generating datasets with some configuration"""
import numpy as np

from .selector import Selector
from sklearn.datasets import make_classification


class DatasetBuilder:
    def __init__(self, init_classify: dict, selector: Selector):
        """
        Create a dataset builder class with some configuration, that can be reused to create similar datasets.
        To generate from the same initial distribution, but using a different selection each time,
        pass an integer for "random_state" in init_classify.

        :param init_classify: parameters for sklearn.dataset.make_classification for the initial dataset
        :param selector: selector object that splits the initial set into source, global and target
        """
        self.init_classify = init_classify
        self.selector = selector

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a new source, global and test dataset using the configuration from initialization.

        :return:  tuple (xs, ys, xg, yg, xt, yt), where s=source, g=global, t=target.
        """
        x, y = make_classification(**self.init_classify)
        xs, ys, xg, yg, xt, yt = self.selector.select(x, y)
        return xs, ys, xg, yg, xt, yt

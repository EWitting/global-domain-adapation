"""Contains builder class for generating datasets with some configuration for concept shift"""
import numpy as np

from .shifter import Shifter
from .selector import DomainSelector
from sklearn.datasets import make_classification


class ConceptShiftDataBuilder:
    def __init__(self, init_classify: dict, shifter: Shifter, selector: DomainSelector):
        """
        Create a dataset builder class with some configuration, that can be reused to create similar datasets.
        To generate from the same initial distribution, but using a different selection each time,
        pass an integer for "random_state" in init_classify.

        :param init_classify: parameters for sklearn.dataset.make_classification for the initial dataset
        :param shifter: Shifter object that creates multiple domains by applying random transformations
        :param selector: DomainSelector that selects with bias, based on domains as created by shifter
        """
        self.init_classify = init_classify
        self.shifter = shifter
        self.selector = selector

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a new source, global and test dataset using the configuration from initialization.

        :return:  tuple (xs, ys, xg, yg, xt, yt), where s=source, g=global, t=target.
        """
        x, y = make_classification(**self.init_classify)
        x, y, domain = self.shifter.shift(x, y)
        xs, ys, xg, yg, xt, yt = self.selector.select(x, y, domain)
        return xs, ys, xg, yg, xt, yt

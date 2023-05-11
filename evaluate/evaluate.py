"""Contains utility functions that can be used for evaluating a dataset and DA model's performance"""

from sklearn.metrics import accuracy_score
import numpy as np


def analyze_data(dataset) -> dict:
    """Analyze properties of a dataset using some directly computable statistical metrics, without ML.
    :param dataset: tuple (xs, ys, xg, yg, xt, yt) where s=source, g=global, t=target.
    :returns dict with metrics
    """
    res = dict()
    xs, ys, xg, yg, xt, yt = dataset
    for x, y, name in [
            (xg, yg, 'global'),
            (xs, ys, 'source'),
            (xt, yt, 'target')]:
        res[f'num-{name}'] = x.shape[0]
        res[f'uniqueness-{name}'] = np.unique(x, axis=0).shape[0] / x.shape[0]
        res[f'class-marginal-{name}'] = np.mean(y)
    return res


def evaluate_model(dataset, model_builder, fit_params: dict) -> dict:
    """Evaluate a domain adaptation model on a dataset, with source only, target only, and global.
    Prints results to console.

    :param dataset: (tuple of Xs, ys, Xg, yg, Xt, yt) data and labels for source, global and target sets
    :param model_builder: (() -> BaseAdaptDeep model) function that returns a new model every call
    :param fit_params: parameters like epoch, batch size etc. for `model.fit`
    :returns dict with accuracies 'acc-{name}' for name in {source, target, global}
    """

    res = dict()
    xs, ys, xg, yg, xt, yt = dataset
    for name, target in [
        ("source (supervised)", xs),
        ("target (domain adaptation)", xt),
        ("global (ours)", xg)]:
        print(f"Evaluating {name}...")
        acc = _eval_single_model(model_builder(), xs, ys, target, xt, yt, fit_params)
        res[f'acc-{name}'] = acc
        print(f"\tDone. Accuracy = {acc}")
    return res


def _eval_single_model(model, xs, ys, xt_train, xt_true, yt, fit_params):
    """Find the accuracy of a DA trained on a given target set, on a true set."""
    model.fit(xs, ys, xt_train, **fit_params)
    yt_pred = model.predict(xt_true)
    return accuracy_score(yt, yt_pred > 0.5)

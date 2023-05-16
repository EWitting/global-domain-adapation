"""Contains utility functions that can be used for evaluating a dataset and DA model's performance"""

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np


def acc_slope(acc: np.ndarray, last_portion: float = 0.05, preceding_portion: float = 0.05) -> float:
    """Compute an indication of convergence, by the slope of the accuracy.
    Averaged over two trail proportions of the acc curve.
    Values near 0 indicate convergence, positive values indicate that progress is still being made."""
    array_length = len(acc)
    last_num = max(1, int(last_portion * array_length))
    preceding_num = max(1, int(preceding_portion * array_length))

    last_samples = acc[-last_num:]
    preceding_samples = acc[-(last_num + preceding_num):-last_num]

    return (np.mean(last_samples) - np.mean(preceding_samples)) / (0.5*(last_num + preceding_num))


def analyze_data(dataset) -> dict:
    """Analyze properties of a dataset using some directly computable statistical measures, without ML.
    :param dataset: tuple (xg, yg, xs, ys, xt, yt) where s=source, g=global, t=target.
    :returns dict with metrics
    """
    res = dict()
    xg, yg, xs, ys, xt, yt = dataset
    for x, y, name in [
            (xg, yg, 'global'),
            (xs, ys, 'source'),
            (xt, yt, 'target')]:
        res[f'num-{name}'] = x.shape[0]
        res[f'uniqueness-{name}'] = np.unique(x, axis=0).shape[0] / x.shape[0]
        res[f'class-marginal-{name}'] = np.mean(y)
    return res


def evaluate_deep(dataset, model_builder, fit_params: dict) -> dict:
    """Evaluate a domain adaptation model on a dataset.
    Trains and evaluates the model multiple times with various combinations of domains.
    Can be used to both evaluate the efficiency of the DA with respect to baselines,
    and give insight into the dataset's characteristics and separability.

    :param dataset: (tuple of xg, yg, xs, ys, xt, yt) data and labels for source, global and target sets
    :param model_builder: (() -> BaseAdaptDeep model) function that returns a new model every call
    :param fit_params: parameters like epoch, batch size etc. for `model.fit`
    :returns dictionary with all computed results
    """

    xg, yg, xs, ys, xt, yt = dataset
    domains = {
        's': {'x': xs, 'y': ys},
        'g': {'x': xg, 'y': yg},
        't': {'x': xt, 'y': yt},
    }

    # first train models
    metrics = dict()
    models = dict()
    pbar = tqdm([
        ('s', 's'),
        ('g', 'g'),
        ('t', 't'),
        ('s', 't'),
        ('s', 'g')])
    for source, target in pbar:
        name = f'{source}-only' if source == target else f'{source}->{target}'
        pbar.set_description(f"Training model '{name}'")
        models[name] = model_builder().fit(
            domains[source]['x'],
            domains[source]['y'],
            domains[target]['x'], **fit_params)

        # compute a convergence indication from the history
        if hasattr(models[name], 'history_'):
            indication = acc_slope(models[name].history_['acc'])
            metrics[f'{name}-convergence-acc-slope'] = indication
        pbar.set_description("Finished training")

    # then evaluate accuracy on different test sets (not every combination is used)
    pbar = tqdm([
        ('s-only', 's'),
        ('g-only', 'g'),
        ('t-only', 't'),
        ('s-only', 't'),
        ('g-only', 't'),
        ('s->t', 't'),
        ('s->g', 't')  # <-- This is the intended usage in real situations
    ])
    for model, test in pbar:
        name = f"{model}-acc-on-{test}"
        pbar.set_description(f"Evaluating model '{model}' on '{test}'")
        x = domains[test]['x']
        y = domains[test]['y']
        y_pred = models[model].predict(x)
        acc = accuracy_score(y, y_pred > 0.5)
        metrics[name] = acc
        pbar.set_description("Finished evaluating")

    # for additional information, train models to classify which domain a point belongs to
    # can be used directly as a distance metric between distributions, for a certain hypothesis class (model type)
    # called proxy A-distance (Ben-David et al., A theory of learning from different domains, 2010)
    # halved so that the range is in [0,1], (assuming classification accuracy is never below 50%)
    pbar = tqdm([('s', 'g'), ('s', 't'), ('g', 't')])
    for domain_a, domain_b in pbar:
        name = f"half-A-dist-{domain_a}-{domain_b}"
        pbar.set_description(f"Estimate proxy A-distance between '{domain_a}' and '{domain_b}'")

        # combine datasets, and use domain as label, permutes randomly
        x = np.concatenate(
            [domains[domain_a]['x'],
             domains[domain_b]['x']],)
        y = np.concatenate([np.full(domains[domain_a]['x'].shape[0], 0),
                           np.full(domains[domain_b]['x'].shape[0], 1)])
        p = np.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        model = model_builder().fit(x, y, x, **fit_params)
        y_pred = model.predict(x)
        acc = accuracy_score(y, y_pred > 0.5)
        metrics[name] = 2*acc - 1  # equiv to 0.5 * a_dist = 0.5 * 2*(1-2*err)
        pbar.set_description("Estimated distance")
    return metrics

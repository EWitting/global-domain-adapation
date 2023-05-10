"""Contains utility functions that can be used for evaluating a DA model's performance"""

from sklearn.metrics import accuracy_score

def evaluate(dataset, model_builder, fit_params):
    """Evaluate a domain adaptation model on a dataset, with source only, target only, and global.
    Prints results to console.

    Args:
        dataset (tuple of Xs, ys, Xg, yg, Xt, yt):
          data and labels for source, global and target sets

        model_builder (() -> BaseAdaptDeep model):
          function that returns a new model every call

        fit_params (dict):  
            parameters like epoch, batch size etc for model.fit
    """

    Xs, ys, Xg, _, Xt, yt = dataset

    for name, target in [
            ("source (supervised)", Xs),
            ("target (domain adaptation)", Xt),
            ("global (ours)", Xg)]:
        print(f"Evaluating {name}...")
        acc = _eval_single_model(model_builder(), Xs, ys, target, Xt, yt, fit_params)
        print(f"\tDone. Accuracy = {acc}")


def _eval_single_model(model, Xs, ys, Xt_train, Xt_true, yt, fit_params):
    """Find the accuraccy of a DA trained on a given target set, on a true set."""
    model.fit(Xs, ys, Xt_train, **fit_params)
    yt_pred = model.predict(Xt_true)
    return accuracy_score(yt, yt_pred > 0.5)

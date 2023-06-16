import os

import numpy as np
import optuna
from adapt.feature_based import DANN

from experiment.presets.bias import bias_names
from experiment.presets.param import auto_param_gen, dann_param_gen
from models.autoencoder import Autoencoder
from util.batch import batch_eval_single

PREFIX = "v4"

FIT_PARAMS = dict(epochs=20,
                  batch_size=16,
                  verbose=0)


def objective(path, model_cls, param_opt, source_domain, target_domain):
    """Generate an objective function for the current study.
    Requires a param_opt function that returns a model_params dict, using trial.suggest..."""
    def _objective(trial):
        model_params = param_opt(trial)
        acc = batch_eval_single(path, model_cls, model_params, FIT_PARAMS, source_domain, target_domain)
        return np.mean(acc)

    return _objective


if __name__ == "__main__":
    """Run hyperparameter optimization on every model and bias type, for every configuration.
    Computes t-only as supervised. Stores result in Optuna database, with PREFIX and identifier in study name."""

    # For each DDA method
    for model, param_generator, n_trials in [
            (Autoencoder, auto_param_gen, 20),
            (DANN, dann_param_gen, 10)]:

        # For each shift type and amount
        for bias in bias_names:
            store_path = os.path.join(os.getcwd(), '../results', PREFIX, bias)

            # For the supervised/adapt-to-target/adapt-to-global configurations
            for source, target, name in [
                    ('t', 't', 't-only'),
                    ('s', 't', 's->t'),
                    ('s', 'g', 's->g')]:

                # Create an optuna study, with a custom objective function that optimizes this configuration
                study_name = f"{PREFIX}-{bias}-{model.__name__}-{name}"
                obj = objective(store_path, model, param_generator, source, target)
                study = optuna.create_study(
                        storage="sqlite:///../db.sqlite3",
                        study_name=study_name,
                        direction='maximize',
                        load_if_exists=False)
                study.optimize(obj, n_trials=n_trials)

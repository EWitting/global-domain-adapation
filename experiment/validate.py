import os

import optuna
from adapt.feature_based import DANN

from experiment.presets.bias import bias_names
from experiment.presets.param import auto_param_gen, dann_param_gen
from models.autoencoder import Autoencoder
from util.batch import batch_eval

PREFIX = "v4"

FIT_PARAMS = dict(epochs=30,
                  batch_size=16,
                  verbose=0)

if __name__ == "__main__":
    """Run the evaluation framework for every model and bias type, for every configuration. Using optimized parameters.
    Loads parameters from Optuna database, based on PREFIX and identifier in study name. Reuses t-only for s-only.
    Runs on the validation data sets."""

    # For each DDA method
    for model, param_generator in [
            (Autoencoder, auto_param_gen),
            (DANN, dann_param_gen)]:

        # For each shift type and amount
        for bias in bias_names:
            print(f"Evaluating {model.__name__} on {bias}")

            # Load optimal hyperparameters for each configuration (reuses t-only for supervised s-only config)
            model_params = dict()
            for eval_name, optim_name in [
                    ('s-only', 't-only'),
                    ('s->g', 's->g'),
                    ('s->t', 's->t'),
                    ('t-only', 't-only')]:
                # Load optuna study by name and use best_trial for parameters
                study_name = f"{PREFIX}-{bias}-{model.__name__}-{optim_name}"
                study = optuna.load_study(study_name=study_name, storage="sqlite:///../db.sqlite3")
                model_params[eval_name] = param_generator(study.best_trial)

            # Run these four configurations on automatically stores results on disk
            store_path = os.path.join(os.getcwd(), '../results', PREFIX, f"{bias}_val")
            batch_eval(store_path, model, model_params, FIT_PARAMS,
                       train_split=.7, multi_param=True, identifier=model.__name__)

import os

import numpy as np
import optuna
from adapt.feature_based import DANN
from tensorflow.keras.optimizers.legacy import Adam

from util.batch import batch_eval_single

store_path = os.path.join(os.getcwd(), 'results', 'batch_quick')


def objective(trial):
    model_params = dict(loss="bce",
                        optimizer=Adam(0.001, beta_1=0.5),
                        lambda_=trial.suggest_float('lambda', 0.6, 1.0),
                        metrics=["acc"], random_state=0)
    fit_params = dict(epochs=25,
                      batch_size=16,
                      verbose=0)
    acc = batch_eval_single(store_path, DANN, model_params, fit_params, 's', 'g')
    return np.mean(acc)


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name='dann_overlap_global',
    direction='maximize',
    load_if_exists=True)
study.optimize(objective, n_trials=10)

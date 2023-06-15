import os

import numpy as np
import optuna
from adapt.feature_based import DANN, MDD
from models.autoencoder import Autoencoder
from tensorflow.keras.optimizers.legacy import Adam

from util.batch import batch_eval_single

model_cls = Autoencoder
batch_name = "v3_cov_strong"
target_domain = "t"
store_path = os.path.join(os.getcwd(), 'results', batch_name)


def objective(trial):
    # model_params = dict(loss="bce",
    #                     optimizer=Adam(0.001, beta_1=0.5),
    #                     lambda_=trial.suggest_float('lambda', 0.0, 5.0),
    #                     # gamma=trial.suggest_float('gamma', 3.5, 6),
    #                     metrics=["acc"], random_state=0)
    # fit_params = dict(epochs=25,
    #                   batch_size=16,
    #                   verbose=0)
    model_params = dict(input_dim=5, encoder_dim=3,
                        aux_classifier_weight=trial.suggest_float('aux_class_weight', 0.0, 4.0),
                        mmd_weight=trial.suggest_float('mmd_weight', 0.0, 5.0),
                        mmd_beta=trial.suggest_float('mmd_beta', 0.0, 5.0))
    fit_params = dict(epochs=15,
                      batch_size=16,
                      verbose=0)
    acc = batch_eval_single(store_path, model_cls, model_params, fit_params, 's', target_domain)
    return np.mean(acc)


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name=f"{batch_name}-{model_cls.__name__}-{target_domain}",
    direction='maximize',
    load_if_exists=True)
study.optimize(objective, n_trials=20)

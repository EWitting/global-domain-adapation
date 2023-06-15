from tensorflow.keras.optimizers.legacy import Adam


def auto_param_opt(trial):
    return dict(input_dim=5, encoder_dim=3,
                aux_classifier_weight=trial.suggest_float('aux_class_weight', 0.0, 10.0),
                mmd_weight=trial.suggest_float('mmd_weight', 0.0, 10.0),
                mmd_beta=trial.suggest_float('mmd_beta', 0.0, 10.0))


def dann_param_opt(trial):
    return dict(loss="bce",
                optimizer=Adam(0.001, beta_1=0.5),
                lambda_=trial.suggest_float('lambda', 0.0, 10.0),
                metrics=["acc"], random_state=0)

from evaluate.evaluate import evaluate

from adapt.feature_based import DANN
from tensorflow.keras.optimizers.legacy import Adam

from datagen.datagen import make_moons_triplet


def model_builder():
    """Returns a new DANN model every call"""
    return DANN(loss="bce", optimizer=Adam(0.001, beta_1=0.5),
                                 lambda_=1, metrics=["acc"], random_state=0)

fit_params = dict(epochs=800, batch_size=34, verbose=0)
dataset = make_moons_triplet(force_balance=True)

evaluate(dataset, model_builder, fit_params)

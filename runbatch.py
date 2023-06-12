import os

from adapt.feature_based import DANN, ADDA, MDD
from tensorflow.keras.optimizers.legacy import Adam


from util.batch import batch_eval
from models.autoencoder import Autoencoder, default_classifier

batch_path = os.path.join(os.getcwd(), 'results', 'v2_concept_weak')

#%% configure and evaluate model
model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=.8, metrics=["acc"], random_state=0,
                    task=default_classifier())
fit_params = dict(epochs=25,
                  batch_size=16,
                  verbose=0)
batch_eval(batch_path, DANN, model_params, fit_params, identifier="DANN")

# #%% MDD
# mdd_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=.1, gamma=4, metrics=["acc"], random_state=0)
# batch_eval(batch_path, MDD, mdd_params, fit_params, identifier="MDD")

#%% Enc
model_params = dict(input_dim=5, encoder_dim=3,
                    aux_classifier_weight=5.0,
                    mmd_weight=.75,
                    mmd_beta=3.5)
fit_params = dict(epochs=15,
                  batch_size=16,
                  verbose=0)
batch_eval(batch_path, Autoencoder, model_params, fit_params, identifier="auto")

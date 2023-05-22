import os

from adapt.feature_based import DANN, ADDA, MDD
from tensorflow.keras.optimizers.legacy import Adam

from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from util.batch import batch_generate, batch_eval

batch_path = os.path.join(os.getcwd(), 'results', 'batch_quick')

#%% configure and generate datasets
init_classification = dict(n_samples=1000, n_features=10, n_informative=10,
                           n_repeated=0, n_redundant=0, n_clusters_per_class=5)
shifter = Shifter(n_domains=4, rot=.15, trans=2, scale=.2)
selector = DomainSelector(n_global=200, n_source=200, n_target=200, n_domains_source=1, n_domains_target=1)
builder = ConceptShiftDataBuilder(init_classification, shifter, selector)

batch_generate(builder, 5, batch_path)

#%% configure and evaluate model
model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=1, metrics=["acc"], random_state=0)
fit_params = dict(epochs=300, batch_size=100, verbose=0)
#
batch_eval(batch_path, DANN, model_params, fit_params, identifier="DANN")
batch_eval(batch_path, MDD, model_params, fit_params, identifier="MDD")

model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), metrics=["acc"], random_state=0)
batch_eval(batch_path, ADDA, model_params, fit_params, identifier="ADDA")

import os

import pandas as pd
from adapt.feature_based import DANN
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorflow.keras.optimizers.legacy import Adam

from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from util.batch import batch_generate, batch_eval, batch_load_eval

# configure dataset generation
init_classification = dict(n_samples=1000, n_features=10, n_informative=10,
                           n_repeated=0, n_redundant=0, n_clusters_per_class=5)
shifter = Shifter(n_domains=2, rot=.15, trans=2, scale=.2)
selector = DomainSelector(n_global=200, n_source=200, n_target=200, n_domains_source=1, n_domains_target=1)

# configure model
model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=1, metrics=["acc"], random_state=0)
fit_params = dict(epochs=300, batch_size=34, verbose=0)

builder = ConceptShiftDataBuilder(init_classification, shifter, selector)

batch_path = os.path.join(os.getcwd(), 'results', 'experiment_batch')
batch_generate(builder, 10, batch_path)
batch_eval(batch_path, DANN, model_params, fit_params)

# process resulting dataframe
results = batch_load_eval(batch_path)
acc = results.loc[:, results.columns.str.contains('acc-on-t')]
acc = acc.rename(columns=lambda col: col.split('metrics.')[1].split('-acc-on-t')[0])
perc_formatter = FuncFormatter(lambda y, _: '{:.0%}'.format(y))

# plot box plots of all accuracies, sorted by median
sorted_columns = acc.median().sort_values()
sorted_acc = acc[sorted_columns.index]
sorted_acc.boxplot()
plt.gca().yaxis.set_major_formatter(perc_formatter)
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# plot adaptation of s->g and s->t
adaptation_g = (acc['s->g'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
adaptation_t = (acc['s->t'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
data = pd.DataFrame({'adaptation_g': adaptation_g, 'adaptation_t': adaptation_t})
data.boxplot()
plt.gca().yaxis.set_major_formatter(perc_formatter)
plt.ylabel("Adaptation between s-only and t-only (%)")
plt.show()

# plot adaptation of s->g where s->t is used as upper bound
adaptation_g_rel_t = (acc['s->g'] - acc['s-only']) / (acc['s->t'] - acc['s-only'])
data2 = pd.DataFrame({'Relative performance': adaptation_g_rel_t})
data2.boxplot()
plt.gca().yaxis.set_major_formatter(perc_formatter)
plt.ylabel("Adaptation between s-only and s->t (%)")
plt.show()

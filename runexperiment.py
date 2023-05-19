from adapt.feature_based import MDD, DANN
from tensorflow.keras.optimizers.legacy import Adam

from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from datagen.conceptshift.builder import ConceptShiftDataBuilder

from util.run import run_experiment, run_eval, run_generate

# configure dataset generation
init_classification = dict(n_samples=1000, n_features=10, n_informative=10,
                           n_repeated=0, n_redundant=0, n_clusters_per_class=5)
shifter = Shifter(n_domains=2, rot=.15, trans=2, scale=.2)
selector = DomainSelector(n_global=200, n_source=200, n_target=200, n_domains_source=1, n_domains_target=1)

# configure model
model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=1, metrics=["acc"], random_state=0)
fit_params = dict(epochs=300, batch_size=34, verbose=0)

builder = ConceptShiftDataBuilder(init_classification, shifter, selector)
run_generate(builder, 'experiment')

deep_metrics = run_eval('experiment', DANN, model_params, fit_params, DANN.__name__)
keys = ('t-only-acc-on-t',
        's->t-acc-on-t',
        's->g-acc-on-t',
        'g-only-acc-on-t',
        's-only-acc-on-t')
selection = [(k, v) for k, v in deep_metrics.items() if k in keys]
selection.sort(key=lambda x: -x[1])
print(f"Results for {model.__name__}")
for key, value in selection:
    print(f"{key[:-9]:6} = {deep_metrics[key]:.3f}")
# run_experiment(builder, MDD, model_params, fit_params, name="experiment")

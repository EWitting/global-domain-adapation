from evaluate.evaluate import evaluate_deep

from adapt.feature_based import DANN
from tensorflow.keras.optimizers.legacy import Adam

from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.visualize import visualize_shift2d
from evaluate.evaluate import analyze_data
from storage.storage import Store

# configure dataset generation
init_classification = dict(n_samples=500, n_features=2, n_informative=2, n_repeated=0, n_redundant=0)
shifter = Shifter(n_domains=2, rot=.25, trans=1, scale=2)
selector = DomainSelector(n_global=100, n_source=100, n_target=100, n_domains_source=1, n_domains_target=1)

builder = ConceptShiftDataBuilder(init_classification, shifter, selector)

# generate and analyze dataset
data = builder.generate()
store = Store.new("experiment", overwrite=True)
store.save_data(*data)

data = store.load_data()
data_metrics = analyze_data(data)
for key, value in sorted(data_metrics.items(), key=lambda x: x[0]):
    print(f"{key}: {data_metrics[key]}")

visualize_shift2d(*data)

# configure model
model_params = dict(loss="bce", optimizer=Adam(0.001, beta_1=0.5), lambda_=1, metrics=["acc"], random_state=0)
fit_params = dict(epochs=400, batch_size=34, verbose=0)

# evaluate model with dataset
deep_metrics = evaluate_deep(data, lambda: DANN(**model_params), fit_params)
for key in deep_metrics:
    print(f"{key}: {deep_metrics[key]:.2f}")

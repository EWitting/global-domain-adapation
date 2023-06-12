import os


from datagen.covshift.builder import CovShiftBuilder
from datagen.covshift.selector import FeatureSelector
from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from util.batch import batch_generate, batch_eval

batch_path = os.path.join(os.getcwd(), 'results', 'v2_concept_weak')

init_classification = dict(n_samples=5000, n_features=5, n_informative=3,
                           n_repeated=0, n_redundant=2, n_clusters_per_class=4)

shifter = Shifter(n_domains=4, rot=0, trans=1, scale=0)
selector = DomainSelector(n_global=1000, n_source=1000, n_target=1000, n_domains_source=1, n_domains_target=1)
builder = ConceptShiftDataBuilder(init_classification, shifter, selector)

# selector = FeatureSelector(n_global=1000, n_source=1000, n_target=1000, source_scale=.75, target_scale=.75, bias_dist=3)
# builder = CovShiftBuilder(init_classification, selector)

batch_generate(builder, 5, batch_path)

import os

from util.batch import batch_load_eval
from util.plot import plot_target_acc_box, plot_adaptation, plot_relative_adaptation

batch_path = os.path.join(os.getcwd(), 'results', 'batch_4_domains')

# process resulting dataframe
results = batch_load_eval(batch_path)

# plot box plots of all accuracies, sorted by median
plot_target_acc_box(results)

# plot adaptation of s->g and s->t
plot_adaptation(results)

# plot adaptation of s->g where s->t is used as upper bound
plot_relative_adaptation(results)

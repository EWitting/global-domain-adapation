import os

from util.batch import batch_load_eval
from util.plot import plot_target_acc_box, plot_adaptation, plot_relative_adaptation

batch_path = os.path.join(os.getcwd(), 'results', 'temp')

# process resulting dataframe
results = batch_load_eval(batch_path)

# group by identifier and make separate plots
results = results.groupby('identifier')
for name, data in results:
    name = str(name)
    if name == "auto":
        name = "Autoenc."

    title = f"Overfit data ({name})"
    plot_target_acc_box(data, title)

    # plot_adaptation(data, title)


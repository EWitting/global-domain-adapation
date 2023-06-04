import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

FORMAT_PERC = FuncFormatter(lambda y, _: '{:.0%}'.format(y))

DISPLAY_NAMES = {
    "s-only": "Source-only",
    "g-only": "Global-only",
    "t-only": "Target-only",
    "s->g": "Adapt-to-global",
    "s->t": "Adapt-to-target"
}


def plot_target_acc_box(results: pd.DataFrame, title: str) -> None:
    """
    Plot for each model the target accuracy boxplot.
    :param results: dataframe in the format returned by batch_load_eval
    :param title
    """
    cols = ['s-only', 's->g', 's->t', 't-only']

    acc = _process_target_acc(results)
    acc = acc[cols]
    acc = acc.rename(columns=DISPLAY_NAMES)
    acc.boxplot()
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.ylim(bottom=0.0, top=1)
    plt.tight_layout()
    plt.show()


def plot_adaptation(results: pd.DataFrame, title: str = None) -> None:
    """
    For adapt-to-global, and adapt-to-target,
    plot the percentage that they reach between source-only and target-only.
    :param results: dataframe in the format returned by batch_load_eval
    :param title
    """
    acc = _process_target_acc(results)
    adaptation_g = (acc['s->g'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
    adaptation_t = (acc['s->t'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
    data = pd.DataFrame({DISPLAY_NAMES['s->g']: adaptation_g,
                         DISPLAY_NAMES['s->t']: adaptation_t})
    data.boxplot(showfliers=False)
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Adaptation (%)")
    plt.tight_layout()
    plt.show()


def plot_relative_adaptation(df: pd.DataFrame, title: str = None) -> None:
    acc = _process_target_acc(df)
    adaptation_g_rel_t = (acc['s->g'] - acc['s-only']) / (acc['s->t'] - acc['s-only'])
    data2 = pd.DataFrame({DISPLAY_NAMES['s->g']: adaptation_g_rel_t})
    data2.boxplot(showfliers=False)
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Relative Adaptation(%)")
    plt.tight_layout()
    plt.show()


def _process_target_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only target acc columns and rename to 'x->y' or 'x-only' format.
    Sorts columns by median."""
    acc = df.loc[:, df.columns.str.contains('acc-on-t')]
    acc = acc.rename(columns=lambda col: col.split('metrics.')[1].split('-acc-on-t')[0])
    sorted_columns = acc.median().sort_values()
    return acc[sorted_columns.index]




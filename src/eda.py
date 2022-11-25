import pandas as pd
import matplotlib.pyplot as plt
import os


def build_grouped_boxplots(df, group_feature, agg_feature, group_custom_order=None,
                           plot_title=None, xlabel=None, ylabel=None, caption=None, file_name=None):
    """Build a boxplot grouped by one feature for one aggregation feature."""

    groupdf = df.groupby([group_feature]).agg({agg_feature: lambda x: list(x)})
    groupdf = groupdf.reset_index().set_index(group_feature)
    groupdf = groupdf.sort_values(by=[group_feature], key=lambda x: x.map(group_custom_order))
    groupdict = groupdf.to_dict()[agg_feature]

    fig, ax = plt.subplots()
    bp = ax.boxplot(groupdict.values(), sym='.', medianprops=dict(color='tab:green'))
    ax.tick_params(axis='both', which='major', labelsize=9)

    if plot_title:
        ax.set_title(plot_title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8)

    if caption:
        ax.text(0.15, -0.1, caption, wrap=False, horizontalalignment='left', fontsize=9)

    if isinstance(group_custom_order, dict):
        ax.set_xticklabels(group_custom_order.keys(), fontdict=dict(fontsize=9))
    plt.grid(True, which='major', color='0.8', linestyle='-')

    if file_name:
        plt.savefig(os.path.join('../figures', file_name), dpi=300)


if __name__ == '__main__':
    time_domain_df = pd.read_csv('../data/time_domain_windows.csv')
    freq_domain_df = pd.read_csv('../data/freq_domain_windows.csv')
    time_freq_domain_df = pd.read_csv('../data/time_freq_domain_windows.csv')
    class_custom_order = {'sedentary': 0, 'stairs': 1, 'walking': 2, 'running': 3}

    # time domain accl. magnitude vs. activity class boxplots
    acclmag_caption = """Box-plot comparing the acceleration magnitude of each activity- 
with class on the horizontal axis and acceleration [g] on the vertical 
axis. As the activity intensity increases so does the reading on the 
accelerometer."""
    build_grouped_boxplots(df=time_domain_df, group_feature='class', agg_feature='td_mean_m',
                           group_custom_order=class_custom_order, plot_title='Acceleration Magnitude vs. Activity Class',
                           xlabel='Activity Class', ylabel='Acceleration Magnitude [g]',
                           caption=acclmag_caption, file_name='acclmag-activity-boxplot.png')


    # time domain energy vs. activity class boxplots
    energy_caption = """Box-plot comparing the energy of each activity- with
class on the horizontal axis and energy on the vertical axis. In general,
as the activity intensity increases so does the energy expended."""
    build_grouped_boxplots(df=time_domain_df, group_feature='class', agg_feature='td_energy',
                           group_custom_order=class_custom_order, plot_title='Energy vs. Activity Class',
                           caption=energy_caption, file_name='energy-activity-boxplot.png')

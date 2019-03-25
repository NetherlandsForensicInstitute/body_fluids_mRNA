"""

Test the validity of the independence assumption.

"""

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from operator import itemgetter
from collections import Counter, OrderedDict

from reimplementation import *


def make_contingency_table(X, y, single, count):
    """
    Calculates the mean of the occurences for each sample marker combination and puts
    these values in a numpy array.

    :param X: numpy array filled with binary data
    :param y: numpy array containing the labels
    :param single: boolean if True calculates the contingency table for the single sample
                   otherwise calculates the contingency table for the mixture sample.
    :return: a contingency table, either for single or mixed cell types.
    """
    if single:
        data_single = np.concatenate((y.T, X), axis=1)

        # split the single samples in groups
        unique_groups = np.unique(y).astype(int)
        splitted_data = np.array([np.array(data_single[data_single[:, 0] == i, :]) for i in unique_groups])

        if count:
            contingency_table_single = np.array([np.sum(splitted_data[i][:, 1::], axis=0) for i in unique_groups])
        else:
            contingency_table_single = np.array([np.mean(splitted_data[i][:, 1::], axis=0) for i in unique_groups])

        return contingency_table_single
    else:
        data_mixture = np.concatenate((y.T, X), axis=1)

        # split the single samples in groups
        unique_groups_mixt, indices = np.unique(y, return_index=True)
        unique_groups_mixt = unique_groups_mixt[np.argsort(indices)]
        splitted_data_mixt = np.array([np.array(data_mixture[data_mixture[:, 0] == i, :]) for i in unique_groups_mixt])

        if count:
            contingency_table_mixt = np.array(
                [np.sum(splitted_data_mixt[i][:, 1::], axis=0) for i in range(len(unique_groups_mixt))])
        else:
            contingency_table_mixt = np.array(
                [np.mean(splitted_data_mixt[i][:, 1::], axis=0) for i in range(len(unique_groups_mixt))])

        return contingency_table_mixt


def equation_values_per_marker(contingency_table_single, contingency_table_mixt, markerindex,
                               mixturesample):
    """
    Collects the relevant proportions for the single samples and mixture samples in a list.

    :param contingency_table_single: 9 x N numpy array with mean value in each cell
    :param contingency_table_mixt: 7 x N numpy array with mean value in each cell
    :param markerindex: the index for the marker for which the result should be returned
    :param mixturesample: str name of the mixture sample (i.e. two body fluids in one sample)
    :return:
    """
    # get relevant single index
    individual_parts = mixturesample.split('+')
    sorted_classes_map = OrderedDict(sorted(classes_map.items(), key=itemgetter(1)))
    single_names = np.array([list(sorted_classes_map.keys())])
    index1 = int(np.where(single_names == individual_parts[0])[1])
    index2 = int(np.where(single_names == individual_parts[1])[1])

    # get relevant mixture index
    sorted_test_map = OrderedDict(sorted(inv_test_map.items()))
    mixture_names = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]
    mixture_names = np.reshape(mixture_names, (7, 1))
    index3 = int(np.where(mixture_names.flatten() == mixturesample)[0])

    P1 = contingency_table_single[index1, markerindex]
    P2 = contingency_table_single[index2, markerindex]
    P1P2 = P1*P2
    proportion_combination = P1 + P2 - P1P2
    proportion_mixture = contingency_table_mixt[index3, markerindex]

    return proportion_combination, proportion_mixture


def calculate_standard_error(proportion, nobs, alpha):
    z = st.norm.ppf(1-(alpha/2))
    se = z * np.sqrt((proportion*(1-proportion)) / nobs)
    return se


def plot_proportions(names):
    fig = plt.figure(figsize=(50, 45))
    plot_legend = [True, False, False, False, False, False, False]

    errors = np.zeros((7, 2, 19))
    n_observations_single = list(n_per_class.values())
    n_observations_combined_single = [
        n_observations_single[0] + n_observations_single[2],
        n_observations_single[1] + n_observations_single[0],
        n_observations_single[2] + n_observations_single[3],
        n_observations_single[3] + n_observations_single[4],
        n_observations_single[3] + n_observations_single[7],
        n_observations_single[4] + n_observations_single[7],
        n_observations_single[7] + n_observations_single[0]
    ]
    sorted_test_map = OrderedDict(sorted(n_per_mixture_class.items()))
    # correct order
    n_observations_mixt = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]

    for idx, name in enumerate(names):
        # get proportions for all marker values
        proportions = np.array([equation_values_per_marker(
            contingency_table_single, contingency_table_mixt, i, name) for i in range(n_features)])
        df_proportions = pd.DataFrame({'expected': proportions[:, 0], 'empirical': proportions[:, 1]})

        # calculate the standard errors so confidence intervals can be plotted
        for mv in range(contingency_table_mixt.shape[1]):
            # single errors
            errors[idx, 0, mv] = calculate_standard_error(
                proportions[:, 0][mv], n_observations_combined_single[idx],
                alpha=0.05
            )

            # mixture errors
            errors[idx, 1, mv] = calculate_standard_error(
                contingency_table_mixt[idx, mv], n_observations_mixt[idx],
                alpha=0.05
            )

        ax = plt.subplot(241+idx)
        df_proportions.plot.bar(width=0.8, ax=ax, legend=None,
                                yerr=errors[idx, :, :], capsize=4,
                                color=['orange', 'mediumblue'],
                                figsize=(34, 12))
        ax.set_xticklabels(list(df.columns))
        ax.set_title(name, fontsize=24)
        plt.ylabel("Proportion", fontsize=18)
        plt.xlabel("Markers", fontsize=18)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=10)
        if idx == len(names)-1:
            plt.legend(loc=9, fontsize=16, bbox_to_anchor=(1.2, 1))
    #plt.show()
    plt.tight_layout()
    plt.savefig("proportions")
    
    return errors


def plot_difference_in_proportions(names):
    fig = plt.figure(figsize=(25, 21))
    plot_legend = [True, False, False, False, False, False, False]
    for idx, name in enumerate(names):
        # get proportions for all marker values
        proportions = np.array([equation_values_per_marker(
            contingency_table_single, contingency_table_mixt, i, name) for i in range(n_features)])
        df_proportions = pd.DataFrame({'combined': np.subtract(proportions[:, 0], proportions[:, 1])})

        color = np.array([np.where(df_proportions > 0, 'mediumblue', 'orange').flatten().tolist()])
        ax = plt.subplot(421+idx)
        df_proportions.plot.bar(width=0.8, ax=ax, legend=plot_legend[idx],
                                color=color)
        ax.set_xticklabels(list(df.columns))
        ax.set_title(name, fontsize=25)
        plt.yticks(np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]))
        plt.legend(fontsize=14)
        plt.ylabel("Absolute difference in proportions")
        plt.xlabel("Markers")
        plt.xticks(fontsize=8, rotation=0)
        plt.yticks(fontsize=10)
    #plt.show()
    plt.tight_layout()
    plt.savefig('difference_proportions')


if __name__ == '__main__':
    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map, inv_classes_map, n_per_class = \
        get_data_per_cell_type(filename='Datasets/Dataset_NFI_removed_degraded_rv.xlsx')
    X_mixtures, y_mixtures, y_mixtures_n_hot, test_map, inv_test_map = read_mixture_data(
        n_single_cell_types-1, n_features, classes_map)
    n_per_mixture_class = Counter(y_mixtures)

    # Create contingency table for single samples
    single_samples = combine_samples(X_raw_singles)
    y_raw_singles = (np.array([y_raw_singles]))
    contingency_table_single = make_contingency_table(
        single_samples, y_raw_singles, single=True, count=False)

    # Create contingency table for mixture samples
    mixture_samples = combine_samples(X_mixtures)
    y_mixtures = (np.array([y_mixtures]))
    contingency_table_mixt = make_contingency_table(
        mixture_samples, y_mixtures, single=False, count=False)

    # Check for each mixture combination and marker whether the following equation holds
    # P_{sample1} + P_{sample2} - P_{sample1}*P_{sample2} = P_{sample1+sample2}
    df, rv = read_df('Datasets/Dataset_mixtures_rv.xlsx', binarize=True)
    unique_groups_mixt, indices = np.unique(y_mixtures, return_index=True)
    sorted_test_map = OrderedDict(sorted(inv_test_map.items()))
    names = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]

    # Plot when interaction term is not subtracted
    errors = plot_proportions(names)

    # Plot the differences between P_{sample1} + P_{sample2} - P_{sample1}*P_{sample2} and P_{sample1+sample2}
    # i.e. abs(P_{sample1} + P_{sample2} - P_{sample1}*P_{sample2} - P_{sample1+sample2})
    plot_difference_in_proportions(names)
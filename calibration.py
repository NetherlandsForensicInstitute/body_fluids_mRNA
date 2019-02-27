"""

Check whether calibration is needed and if so perform calibration.

"""

import pickle
import collections

import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter

from reimplementation import get_data_per_cell_type, read_mixture_data, \
    calculate_lrs, augment_data

from lir.calibration import KDECalibrator


def plot_individual_histograms(y, log_lrs, names):
    """

    :param y:
    :param log_lrs:
    :param names:
    :return:
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']
    for j, j_class in enumerate(set(y)):
        indices_experiments = [k for k in range(len(y)) if y[k] == j_class]
        plt.subplots(2, 5, figsize=(18, 9))
        plt.suptitle(inv_test_map[j_class], y=1.0, fontsize=14)
        for i in range(log_lrs.shape[1]):
            plt.subplot(2, 5, i + 1)
            plt.xlim([-MAX_LR - .5, MAX_LR + .5])
            plt.hist(log_lrs[indices_experiments, i],
                     color=colors[j])
            plt.title(names[0][i])
        plt.tight_layout()
        plt.show()


def plot_reliability_plots(log_lrs, names):
    """

    :param log_lrs:
    :param names:
    :return:
    """
    global i
    plt.subplots(2, 5, figsize=(18, 9))
    length_lrs = log_lrs.shape[0]
    for i in range(log_lrs.shape[1]):
        plt.subplot(2, 5, i + 1)
        plt.plot(np.linspace(-MAX_LR - .5, MAX_LR + .5, length_lrs),
                 np.linspace(-MAX_LR - .5, MAX_LR + .5, length_lrs),
                 color='k')
        plt.plot(sorted(log_lrs[:, i]),
                 np.linspace(-MAX_LR - .5, MAX_LR + .5, length_lrs))
        plt.title(names[0][i])
    #plt.show()
    plt.savefig('reliability_plots.png')


def plot_histograms_of_lrs(log_lrs, y_mixtures_matrix, inv_y_mixtures_matrix):
    """

    :param log_lrs:
    :param y_mixtures_matrix:
    :param inv_y_mixtures_matrix:
    :return:
    """
    # h1 are all the LRs from a mixt. cell types in which a specific cell type exists
    h1s = np.multiply(log_lrs, y_mixtures_matrix)
    # h2 are all the LRs from a mixt. cell types in which the specific cell type is not
    h2s = np.multiply(log_lrs, inv_y_mixtures_matrix)
    plt.subplots(2, 5, figsize=(18, 9))
    for i in range(log_lrs_per_class.shape[1]):
        plt.subplot(2, 5, i + 1)
        plt.hist(h1s[:, i], bins=30, alpha=0.7, color='mediumblue')
        plt.hist(h2s[:, i], bins=30, alpha=0.7, color='orange')
        plt.title(names_single[0][i])
        plt.legend(('h1', 'h2'))
    plt.show()


if __name__ == '__main__':
    MAX_LR = 10
    N_SAMPLES_PER_COMBINATION = 50

    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map,\
        inv_classes_map, n_per_class = get_data_per_cell_type()
    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = \
        read_mixture_data(n_single_cell_types - 1, n_features, classes_map=classes_map)

    mixture_classes_in_classes_to_evaluate = pickle.load(
        open('mixture_classes_in_classes_to_evaluate', 'rb'))

    model = pickle.load(open('mlpmodel', 'rb'))
    log_lrs_per_class = calculate_lrs(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=True)
    # exclude penile
    log_lrs_per_class = log_lrs_per_class[:, :-1]
    lrs_per_class = calculate_lrs(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=False)
    # exclude penile
    lrs_per_class = lrs_per_class[:, :-1]

    # Plot the log_lrs_per_class per sample: histograms
    n_per_mixture_class = collections.Counter(y_mixtures)
    unique_groups_mixt, indices = np.unique(y_mixtures, return_index=True)
    sorted_test_map = collections.OrderedDict(sorted(n_per_mixture_class.items()))
    n_observations_mixt = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]

    sorted_classes_map = collections.OrderedDict(sorted(classes_map.items(), key=itemgetter(1)))
    names_single = np.array([list(sorted_classes_map.keys())])

    #plot_individual_histograms(y_mixtures, log_lrs_per_class, names_single)

    # TODO: check whether the plotted values are correct --> sorted(log_lrs)
    # TODO: check whether correct values are plotted --> transform into bins?
    #plot_reliability_plots(log_lrs_per_class, names_single)

    inv_y_mixtures_matrix = np.ones_like(y_mixtures_matrix) - y_mixtures_matrix
    #plot_histograms_of_lrs(log_lrs_per_class, y_mixtures_matrix, inv_y_mixtures_matrix)

    # Perform calibration
    X_raw_singles_calibrate = pickle.load(open('X_raw_singles_calibrate', 'rb'))
    y_raw_singles_calibrate = pickle.load(open('y_raw_singles_calibrate', 'rb'))
    X_augmented_calibrate, y_augmented_calibrate, y_augmented_matrix_calibrate, \
    mixture_classes_in_single_cell_type = augment_data(
        X_raw_singles_calibrate,
        y_raw_singles_calibrate,
        n_single_cell_types,
        n_features,
        N_SAMPLES_PER_COMBINATION
    )

    calibrator = KDECalibrator()
    calibrator.fit(X_raw_singles_calibrate, y_raw_singles_calibrate)
    calibrated_LRs = calibrator.transform(X_mixtures)








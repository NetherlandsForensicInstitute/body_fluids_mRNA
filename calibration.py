"""

Check whether calibration is needed and if so perform calibration.

http://danielnee.com/tag/reliability-diagram/

"""

import pickle
import collections

import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter

from reimplementation import get_data_per_cell_type, read_mixture_data, \
    calculate_lrs, augment_data

from lir.calibration import KDECalibrator
from lir.pav import plot


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


def plot_lrs_against_expectedlrs(log_lrs, names):
    """

    :param log_lrs:
    :param names:
    :return:
    """
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
        plt.xlabel("Log likelihood ratio")
        plt.ylabel("Expected likelihood ratio")
    plt.show()
    #plt.savefig('reliability_plots.png')


def reliability_curve(y_true, y_score, bins=10, normalize=False):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities whould be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels (0 or 1).

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=False
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is mapped onto 0 and the largest one
        onto 1.


    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.


    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    y_score = np.array(sorted(y_score))
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        if len(y_score[bin_idx]) > 0:
            y_score_bin_mean[i] = y_score[bin_idx].mean()
            empirical_prob_pos[i] = y_true[bin_idx].mean()
        else:
            print("The bin_idx is empty")
            y_score_bin_mean[i] = 0
            empirical_prob_pos[i] = 0
    print(empirical_prob_pos)
    return y_score_bin_mean, empirical_prob_pos


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
    log_scores_per_class = calculate_scores(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=True)
    # exclude penile
    log_scores_per_class = log_scores_per_class[:, :-1]
    lrs_scores_class = calculate_scores(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=False)
    # exclude penile
    lrs_scores_class = lrs_scores_class[:, :-1]

    # Plot the log_lrs_per_class per sample: histograms
    n_per_mixture_class = collections.Counter(y_mixtures)
    unique_groups_mixt, indices = np.unique(y_mixtures, return_index=True)
    sorted_test_map = collections.OrderedDict(sorted(n_per_mixture_class.items()))
    n_observations_mixt = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]

    sorted_classes_map = collections.OrderedDict(sorted(classes_map.items(), key=itemgetter(1)))
    names_single = np.array([list(sorted_classes_map.keys())])

    #plot_individual_histograms(y_mixtures, log_lrs_per_class, names_single)

    # TODO: check whether the plotted values are correct --> sorted(log_lrs)
    plot_lrs_against_expectedlrs(log_scores_per_class, names_single)

    probabilities = (scores_per_class / (1+scores_per_class))
    reliability_scores = {}
    celltypes = sorted(classes_map)
    celltypes.remove("Skin.penile")
    for i, celltype in enumerate(celltypes):
        reliability_scores[celltype] = reliability_curve(
            y_mixtures_matrix[:, i], probabilities[:, i], bins=10)

    plt.figure(0, figsize=(8, 8))
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
    for celltype, (y_score_bin_mean, empirical_prob_pos) in reliability_scores.items():
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                 empirical_prob_pos[scores_not_nan], label=celltype,
                 color='orange')
    plt.ylabel("Empirical probability")
    plt.legend(loc=0)

    inv_y_mixtures_matrix = np.ones_like(y_mixtures_matrix) - y_mixtures_matrix
    #plot_histograms_of_lrs(log_lrs_per_class, y_mixtures_matrix, inv_y_mixtures_matrix)

    # plot PAV plots
    #plot(lrs_per_class[:, 0], np.array(y_mixtures))

    '''
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
    '''







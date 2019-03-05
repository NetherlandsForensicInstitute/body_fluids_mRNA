"""

Check whether calibration is needed and if so perform calibration.
Note that this script assumes that the calibrated model is used!

http://danielnee.com/tag/reliability-diagram/

http://jmetzen.github.io/2014-08-16/reliability-diagram.html

"""

import pickle
import collections

import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter

from reimplementation import get_data_per_cell_type, read_mixture_data, \
    calculate_scores, evaluate_model, combine_samples

from sklearn.calibration import calibration_curve

from lir.calibration import KDECalibrator
from lir.pav import plot
from lir.util import Xn_to_Xy, Xy_to_Xn


def plot_histograms_of_probabilities(h1_h2_probs):
    plt.subplots(2, 5, figsize=(18, 9))
    for i in range(len(h1_h2_probs)):
        plt.subplot(2, 5, i + 1)
        plt.hist(h1_h2_probs[i][0], bins=30, alpha=0.7, color='mediumblue')
        plt.hist(h1_h2_probs[i][1], bins=30, alpha=0.7, color='orange')
        plt.title(str(i))
        plt.legend(('h1', 'h2'))
    plt.show()


def plot_histogram_log_lr(h1_h2_probs):
    bins = np.linspace(-10, 10, 30)
    plt.subplots(2, 5, figsize=(18, 9))
    for i in range(len(h1_h2_probs)):
        log_likrats1 = np.log10(h1_h2_probs[i][0] / (1 - h1_h2_probs[i][0]))
        log_likrats2 = np.log10(h1_h2_probs[i][1] / (1 - h1_h2_probs[i][1]))

        plt.subplot(2, 5, i + 1)
        plt.hist([log_likrats1, log_likrats2], bins=bins, color=['pink', 'blue'],
                 label=['h1', 'h2'])
        plt.legend(loc='upper right')
        plt.title(str(i))
    plt.show()


def plot_reliability_plot(h1_h2_probs, y_matrix, bins=10):
    plt.subplots(2, 5, figsize=(18, 9))
    for i in range(len(h1_h2_probs)):
        h1h2_probs = np.append(h1_h2_probs[i][0], h1_h2_probs[i][1])
        y_true = sorted(y_matrix[:, i], reverse=True)

        y_score_bin_mean, empirical_prob_pos = calibration_curve(
            y_true, h1h2_probs, n_bins=bins)
        print(y_score_bin_mean)
        print(empirical_prob_pos)

        plt.subplot(2, 5, i + 1)
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                 empirical_prob_pos[scores_not_nan],
                 color='red',
                 marker='o',
                 linestyle='-',
                 label=str(i))
        plt.ylabel("Empirical probability")
        plt.legend(loc=0)
    plt.show()


def perform_calibration(h1_h2_probs_calibration):
    for i in range(len(h1_h2_probs_calibration)):
        X, y = Xn_to_Xy(h1_h2_probs_calibration[i][0], h1_h2_probs_calibration[i][1])
        calibrator = KDECalibrator()
        lr1, lr2 = Xy_to_Xn(calibrator.fit_transform(X, y), y)


if __name__ == '__main__':
    pass

    # TODO: Make this work.
    # plot PAV plots
    # plot(lrs_per_class[:, 0], np.array(y_mixtures))







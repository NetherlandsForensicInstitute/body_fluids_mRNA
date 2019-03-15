"""

Check whether calibration is needed and if so perform calibration.

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve

from lir.calibration import KDECalibrator
from lir.util import *

def plot_histograms_of_probabilities(h1_h2_probs, n_bins=100):
    plt.subplots(4, 2, figsize=(18, 36))
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        plt.subplot(4, 2, idx + 1)
        plt.hist(h1_h2_probs[celltype][0], bins=n_bins, alpha=0.7, color='pink')
        plt.hist(h1_h2_probs[celltype][1], bins=n_bins, alpha=0.7, color='blue')
        plt.title(celltype)
        plt.ylabel("Frequency")
        plt.xlabel("Probabilities with n_bins {}".format(n_bins))
        plt.legend(("h1", "h2"), loc=9)
    plt.show()


def plot_histogram_log_lr(h1_h2_probs, n_bins=30, title='before'):

    bins = np.linspace(-10, 10, n_bins)
    plt.subplots(5, 2, figsize=(18, 36))
    plt.suptitle('Histogram of log LRs {} calibration'.format(title), size=16)
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        log_likrats1 = np.log10(h1_h2_probs[celltype][0] / (1 - h1_h2_probs[celltype][0]))
        log_likrats2 = np.log10(h1_h2_probs[celltype][1] / (1 - h1_h2_probs[celltype][1]))

        plt.subplot(5, 2, idx + 1)
        plt.hist([log_likrats1, log_likrats2], bins=bins, color=['pink', 'blue'],
                 label=["h1", "h2"])
        plt.axvline(x=0, color='k', linestyle='-')
        plt.legend(loc='upper right')
        plt.ylabel("Frequency")
        plt.xlabel("log LR with n_bins {}".format(n_bins))
        plt.title(celltype)
    plt.show()
    #plt.savefig('histogram_log_lr')


def plot_reliability_plot(h1_h2_probs, y_matrix, title, bins=10):

    plt.subplots(5, 2, figsize=(18, 36))
    plt.suptitle("Reliability plot {} calibration".format(title), size=16)
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        h1h2_probs = np.append(h1_h2_probs[celltype][0], h1_h2_probs[celltype][1])
        y_true = sorted(y_matrix[:, idx], reverse=True)

        empirical_prob_pos, y_score_bin_mean = calibration_curve(
            y_true, h1h2_probs, n_bins=bins)

        plt.subplot(5, 2, idx + 1)
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                 empirical_prob_pos[scores_not_nan],
                 color='red',
                 marker='o',
                 linestyle='-',
                 label=celltype)
        plt.xlabel("Probabilities with n_bins {}".format(len(empirical_prob_pos)))
        plt.ylabel("Empirical probability")
        plt.legend(loc=9)
    plt.show()
    #plottitle = maintitle.replace(" ", "_").lower()
    #plt.savefig(plottitle)


def calibration_fit(h1_h2_probs, classes_map, Calibrator=KDECalibrator):
    """
    Get a calibrated model for each class based on one vs. all.

    :param h1_h2_probs:
    :param classes_map:
    :param Calibrator:
    :return:
    """
    calibrators_per_class = {}
    for j, celltype in enumerate(sorted(classes_map)):
        h1_h2_probs_celltype = h1_h2_probs[celltype]

        # TODO: Why does it seem like labels are switched?
        X, y = Xn_to_Xy(h1_h2_probs_celltype[0],
                        h1_h2_probs_celltype[1])

        calibrator = Calibrator()
        calibrators_per_class[celltype] = calibrator.fit(X, y)

    return calibrators_per_class


def calibration_transform(h1_h2_probs_test, calibrators_per_class, classes_map):
    """
    Transforms the scores with the calibrated model for the correct class.

    :param h1_h2_probs_test:
    :param calibrators_per_class:
    :param classes_map:
    :return:
    """
    h1_h2_after_calibration = {}
    for celltype in sorted(classes_map):
        h1_h2_probs_celltype_test = h1_h2_probs_test[celltype]
        calibrator = calibrators_per_class[celltype]
        Xtest, ytest = Xn_to_Xy(h1_h2_probs_celltype_test[0],
                                h1_h2_probs_celltype_test[1])
        
        lr1, lr2 = Xy_to_Xn(calibrator.transform(Xtest), ytest)
        probs1 = lr1 / (1 + lr1)
        probs2 = lr2 / (1 + lr2)

        h1_h2_after_calibration[celltype] = (probs2, probs1)

    return h1_h2_after_calibration
        





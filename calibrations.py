"""

Check whether calibration is needed and if so perform calibration.

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve

from lir.calibration import KDECalibrator
from lir.pav import plot
from lir.util import Xn_to_Xy, Xy_to_Xn


def plot_histograms_of_probabilities(h1_h2_probs, n_bins=10):
    plt.subplots(2, 5, figsize=(18, 9))
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        plt.subplot(2, 5, idx + 1)
        plt.hist(h1_h2_probs[celltype][0], bins=n_bins, alpha=0.7, color='pink')
        plt.hist(h1_h2_probs[celltype][1], bins=n_bins, alpha=0.7, color='blue')
        plt.title(celltype)
        plt.ylabel("Frequency")
        plt.xlabel("Probabilities with n_bins {}".format(n_bins))
        plt.legend(("h1", "h2"), loc=9)
    plt.show()


def plot_histogram_log_lr(h1_h2_probs, n_bins=10):
    bins = np.linspace(-10, 10, n_bins)
    plt.subplots(2, 5, figsize=(18, 9))
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        log_likrats1 = np.log10(h1_h2_probs[celltype][0] / (1 - h1_h2_probs[celltype][0]))
        log_likrats2 = np.log10(h1_h2_probs[celltype][1] / (1 - h1_h2_probs[celltype][1]))

        plt.subplot(2, 5, idx + 1)
        plt.hist([log_likrats1, log_likrats2], bins=bins, color=['pink', 'blue'],
                 label=["h1", "h2"])
        plt.axvline(x=0, color='k', linestyle='-')
        plt.legend(loc='upper right')
        plt.ylabel("Frequency")
        plt.xlabel("log LR with n_bins {}".format(n_bins))
        plt.title(celltype)
    plt.show()
    #plt.savefig('histogram_log_lr')


def plot_reliability_plot(h1_h2_probs, y_matrix, bins=10, title='before'):
    maintitle = ''
    if title == 'before':
        maintitle = "Reliability plot before calibration"
    elif title == 'after':
        maintitle = "Reliability after calibration"
    elif title == 'train':
        maintitle = "Reliability train data"
    elif title == 'mixture':
        maintitle = "Reliability mixture data"

    plt.subplots(2, 5, figsize=(18, 9))
    plt.suptitle(maintitle, size=16)
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        h1h2_probs = np.append(h1_h2_probs[celltype][0], h1_h2_probs[celltype][1])
        y_true = sorted(y_matrix[:, idx], reverse=True)

        y_score_bin_mean, empirical_prob_pos = calibration_curve(
            y_true, h1h2_probs, n_bins=bins)

        plt.subplot(2, 5, idx + 1)
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                 empirical_prob_pos[scores_not_nan],
                 color='red',
                 marker='o',
                 linestyle='-',
                 label=celltype)
        plt.xlabel("Log LR with n_bins {}".format(bins))
        plt.ylabel("Empirical probability")
        plt.legend(loc=9)
    #plt.show()
    plottitle = maintitle.replace(" ", "_").lower()
    plt.savefig(plottitle)


def perform_calibration(h1_h2_probs):
    h1_h2_after_calibration = {}
    for idx, celltype in enumerate((h1_h2_probs.keys())):
        # TODO: Correct values in Xn_to_Xy function?
        scores1 = h1_h2_probs[celltype][0] / (1 - h1_h2_probs[celltype][0])
        scores2 = h1_h2_probs[celltype][1] / (1 - h1_h2_probs[celltype][1])
        X, y = Xn_to_Xy(scores1, scores2)
        # TODO: Need all scores in stead of just for one cell type when performing calibration?
        calibrator = KDECalibrator()
        lr1, lr2 = Xy_to_Xn(calibrator.fit_transform(X, y), y)
        # make probabilities
        probs1 = lr1 / (1+lr1)
        probs2 = lr2 / (1+lr2)
        h1_h2_after_calibration[celltype] = (probs1, probs2)

    return h1_h2_after_calibration







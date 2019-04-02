"""

Check whether calibration is needed and if so perform calibration.

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve

from lir.calibration import KDECalibrator
from lir.util import *
from lir.pav import *


def plot_histogram_log_lr(h0_h1_lrs, n_bins=30, title='before', density=None, savefig=None, show=None):

    celltypes = list(h0_h1_lrs.keys())
    plt.subplots(int(len(celltypes)/2), 2, figsize=(9, 9/4*len(celltypes)), sharey='row')
    # plt.suptitle('Histogram of log LRs {} calibration'.format(title), size=16)

    minimum_lik = np.min([min(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
    maximum_lik = np.max([max(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
    outer_lik = max(abs(minimum_lik), abs(maximum_lik))

    for idx, celltype in enumerate(sorted(celltypes)):
        log_likrats0 = h0_h1_lrs[celltype][0]
        log_likrats1 = h0_h1_lrs[celltype][1]

        plt.subplot(int(len(celltypes)/2), 2, idx + 1)
        plt.hist(log_likrats0, density=density, color='orange', label='h0', bins=n_bins, alpha=0.5)
        plt.hist(log_likrats1, density=density, color='blue', label='h1', bins=n_bins, alpha=0.5)
        plt.legend(loc='upper right')
        if title == 'after':
            plt.xlim(-(outer_lik + 0.05), (outer_lik + 0.05))
        plt.ylabel("Density")
        plt.xlabel("10logLR")
        plt.title(celltype, fontsize=16)
        # plt.text(0.2, 0.9, 'N_train = 100,\nN_test = 50,\nN_calibration = 4',
        #          ha='center', va='center', transform=ax.transAxes)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def plot_reliability_plot(h0_h1_probs, y_matrix, title, bins=10, savefig=None, show=None):

    celltypes = list(h0_h1_probs.keys())
    plt.subplots(int(len(celltypes)/2), 2, figsize=(9, 9 / 4 * len(celltypes)))
    plt.suptitle("Reliability plot {} calibration".format(title), size=16)
    for idx, celltype in enumerate(celltypes):
        h0h1_probs = np.append(h0_h1_probs[celltype][0], h0_h1_probs[celltype][1])
        y_true = sorted(y_matrix[:, idx], reverse=True)

        empirical_prob_pos, y_score_bin_mean = calibration_curve(
            y_true, h0h1_probs, n_bins=bins)

        ax = plt.subplot(int(len(celltypes)/2), 2, idx + 1)
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                 empirical_prob_pos[scores_not_nan],
                 color='red',
                 marker='o',
                 linestyle='-',
                 label=celltype)
        plt.xlabel("Probability".format(len(empirical_prob_pos)))
        plt.ylabel("Empirical probability")
        # plt.text(0.8, 0.1, 'N_train = 100,\nN_test = 50,\nN_calibration = 4',
        #          ha='center', va='center', transform=ax.transAxes)
        plt.legend(loc=9)

    if savefig is not None:
        plottitle = "Reliability plot {} calibration lowcalib".format(title).replace(" ", "_").lower()
        plt.tight_layout()
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def calibration_fit(h0_h1_probs, classes_map, Calibrator=KDECalibrator):
    """
    Get a calibrated model for each class based on one vs. all.

    :param h0_h1_probs:
    :param classes_map:
    :param Calibrator:
    :return:
    """
    calibrators_per_class = {}
    for j, celltype in enumerate(sorted(classes_map)):
        h0_h1_probs_celltype = h0_h1_probs[celltype]

        X, y = Xn_to_Xy(h0_h1_probs_celltype[0],
                        h0_h1_probs_celltype[1])

        calibrator = Calibrator()
        calibrators_per_class[celltype] = calibrator.fit(X, y)

    return calibrators_per_class


def calibration_transform(h0_h1_probs_test, calibrators_per_class, classes_map):
    """
    Transforms the scores with the calibrated model for the correct class.

    :param h0_h1_probs_test:
    :param calibrators_per_class:
    :param classes_map:
    :return:
    """
    h0_h1_after_calibration = {}
    for celltype in sorted(classes_map):
        h0_h1_probs_celltype_test = h0_h1_probs_test[celltype]
        calibrator = calibrators_per_class[celltype]
        Xtest, ytest = Xn_to_Xy(h0_h1_probs_celltype_test[0],
                                h0_h1_probs_celltype_test[1])
        
        lr0, lr1 = Xy_to_Xn(calibrator.transform(Xtest), ytest)
        probs0 = lr0 / (1 + lr0)
        probs1 = lr1 / (1 + lr1)

        h0_h1_after_calibration[celltype] = (probs0, probs1)

    return h0_h1_after_calibration


def plot_pav(lrs_before, lrs_after, y, classes_map, show_scatter=True, on_screen=False, path=None):
    """
    Plots pav plots for all cell types before and after calibration.

    :param lrs_before:
    :param lrs_after:
    :param y:
    :param show_scatter:
    :param on_screen:
    :param path:
    :return:
    """

    celltypes = lrs_before.keys()
    fig, axs = plt.subplots(len(list(celltypes)), 2, figsize=(9, int(9 / 2 * len(list(celltypes)))))
    for idx, celltype in enumerate(sorted(celltypes)):
        i_celltype = classes_map[celltype]
        lrs_celltype_before = np.append(lrs_before[celltype][0], lrs_before[celltype][1])
        lrs_celltype_after = np.append(lrs_after[celltype][0], lrs_after[celltype][1])

        llrs_celltype_before = np.log10(lrs_celltype_before)
        llrs_celltype_after = np.log10(lrs_celltype_after)

        # Plot before
        pav_before = PavLogLR()
        pav_llrs = pav_before.fit_transform(llrs_celltype_before, y[:, i_celltype])

        all_llrs = np.concatenate([llrs_celltype_before, pav_llrs])
        all_llrs[all_llrs == -np.inf] = 0
        all_llrs[all_llrs == np.inf] = 0
        xrange = [all_llrs.min() - .5, all_llrs.max() + .5]

        axs[i_celltype, 0].axis('equal')
        axs[i_celltype, 0].axis(xrange + xrange)
        axs[i_celltype, 0].plot(xrange, xrange)  # rechte lijn door de oorsprong

        pav_x = np.arange(*xrange, .01)
        axs[i_celltype, 0].set_title(celltype + "\n" + " before calibration", fontsize=12)
        axs[i_celltype, 0].plot(pav_x, pav_before.transform(pav_x))  # pre-/post-calibrated lr fit
        axs[i_celltype, 0].grid(True, linestyle=':')
        if show_scatter:
            axs[i_celltype, 0].scatter(llrs_celltype_before, pav_llrs)  # scatter plot of measured lrs

        # Plot after
        pav_after = PavLogLR()
        pav_llrs_after = pav_after.fit_transform(llrs_celltype_after, y[:, i_celltype])

        axs[i_celltype, 1].axis('equal')
        axs[i_celltype, 1].axis(xrange + xrange)
        axs[i_celltype, 1].plot(xrange, xrange)  # rechte lijn door de oorsprong

        pav_x = np.arange(*xrange, .01)
        axs[i_celltype, 1].set_title(celltype + "\n" + "after calibration", fontsize=12)
        axs[i_celltype, 1].plot(pav_x, pav_after.transform(pav_x))  # pre-/post-calibrated lr fit
        axs[i_celltype, 1].grid(True, linestyle=':')
        if show_scatter:
            axs[i_celltype, 1].scatter(llrs_celltype_after, pav_llrs_after)  # scatter plot of measured lrs

    fig.text(0.5, 0.001, 'pre-PAVcalibrated 10log(lr)', ha='center', fontsize=14)
    fig.text(0.001, 0.5, 'post-PAVcalibrated 10log(lr)', va='center', rotation='vertical', fontsize=14)

    if on_screen:
        plt.show()
    if path is not None:
        plt.tight_layout()
        plt.savefig(path)

    plt.close(fig)





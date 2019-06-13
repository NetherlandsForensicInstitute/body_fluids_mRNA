"""
Plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve

from rna.analytics import combine_samples
# from rna.lr_system import convert_prob_per_mixture_to_marginal_per_class
from rna.utils import vec2string
from lir import PavLogLR

# TODO: Make this function work (?)
def plot_for_experimental_mixture_data(X_mixtures,
                                       y_mixtures,
                                       y_mixtures_matrix,
                                       inv_test_map,
                                       classes_to_evaluate,
                                       mixtures_in_classes_of_interest,
                                       n_single_cell_types_no_penile,
                                       dists):
    """
    for each mixture category that we have measurements on, plot the
    distribution of marginal LRs for each cell type, as well as for the special
    combinations (eg vaginal+menstrual) also plot LRs as a function of distance
    to nearest data point also plot experimental measurements together with LRs
    found and distance in a large matrix plot

    :param X_mixtures: N_experimental_mixture_samples x N_markers array of
        observations
    :param y_mixtures: N_experimental_mixture_samples array of int mixture labels
    :param y_mixtures_matrix:  N_experimental_mixture_samples x
        (N_single_cell_types + N_combos) n_hot encoding
    :param inv_test_map: dict: mixture label -> mixture name
    :param classes_to_evaluate: list of str, classes to evaluate
    :param mixtures_in_classes_of_interest:  list of lists, specifying for each
        class in classes_to_evaluate which mixture labels contain these
    :param n_single_cell_types_no_penile: int: number of single cell types
        excluding penile skin
    :param dists: N_experimental_mixture_samples iterable of distances to
        nearest augmented data point. Indication of whether the point may be an
        outlier (eg measurement error or problem with augmentation scheme)
    """
    y_prob = model.predict_lrs(X_mixtures)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, mixtures_in_classes_of_interest, classes_map_updated, MAX_LR)

    log_lrs_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
    plt.subplots(3, 3, figsize=(18, 9))
    for i, i_clas in enumerate(set(y_mixtures)):
        indices_experiments = [j for j in range(len(y_mixtures)) if y_mixtures[j] == i_clas]
        plt.subplot(3, 3, i + 1)
        plt.xlim([-MAX_LR - .5, MAX_LR + .5])
        bplot = plt.boxplot(log_lrs_per_class[indices_experiments, :], vert=False,
                            labels=classes_to_evaluate, patch_artist=True)

        for j, (patch, cla) in enumerate(zip(bplot['boxes'], classes_to_evaluate)):
            if j < n_single_cell_types_no_penile:
                # single cell type
                if cla in inv_test_map[i_clas]:
                    patch.set_facecolor('black')
            else:
                # sample 'Vaginal.mucosa and/or Menstrual.secretion'
                for comb_class in cla.split(' and/or '):
                    if comb_class in inv_test_map[i_clas]:
                        patch.set_facecolor('black')
        plt.title(inv_test_map[i_clas])
    plt.savefig('mixtures_boxplot')

    plt.subplots(3, 3, figsize=(18, 9))
    for i in range(y_mixtures_matrix.shape[1]):
        plt.subplot(3, 3, i + 1)
        plt.ylim([-MAX_LR - .5, MAX_LR + .5])
        plt.scatter(
            dists + np.random.random(len(dists)) / 20,
            log_lrs_per_class[:, i],
            color=['red' if iv else 'blue' for iv in y_mixtures_matrix[:, i]],
            alpha=0.1
        )
        plt.ylabel('LR')
        plt.xlabel('distance to nearest data point')
        plt.title(classes_to_evaluate[i])
    plt.savefig('LRs_as_a_function_of_distance')

    plt.figure()
    plt.matshow(
        np.append(
            np.append(X_mixtures, log_lrs_per_class, axis=1),
            np.expand_dims(np.array([d*5 for d in dists]), axis=1),
            axis=1))
    plt.savefig('mixtures binned data and log lrs')


def plot_data(X):
    """
    plots the raw data points

    :param X: N_samples x N_observations_per_sample x N_markers measurements
    """
    plt.matshow(combine_samples(X))
    plt.savefig('single_cell_type_measurements_after_QC')


def plot_histogram_log_lr(lrs, y_nhot, target_classes, label_encoder, n_bins=30,
                          title='before', density=False, savefig=None, show=None):

    loglrs = np.log10(lrs)
    n_target_classes = len(target_classes)

    plt.subplots(int(n_target_classes / 2), 2, figsize=(9, int(9 / 4 * n_target_classes)), sharey='row')
    plt.suptitle('Histogram {} calibration'.format(title))
    for i, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        loglrs1 = np.multiply(loglrs[:, i], np.max(np.multiply(y_nhot, target_class), axis=1))
        loglrs2 = np.multiply(loglrs[:, i], 1-np.max(np.multiply(y_nhot, target_class), axis=1))

        # delete zeros
        loglrs1 = np.delete(loglrs1, np.where(loglrs1 == -0.0))
        loglrs2 = np.delete(loglrs2, np.where(loglrs2 == 0.0))

        plt.subplot(int(n_target_classes / 2), 2, i + 1)
        plt.hist(loglrs1, color='orange', density=density, bins=n_bins, label='h1', alpha=0.5)
        plt.hist(loglrs2, color='blue', density=density, bins=n_bins, label='h2', alpha=0.5)

        plt.title(celltype)
        # if title == 'after':
        #     outer_lik = max(abs(np.min(loglrs)), abs(nqp.max(loglrs)))
        #     plt.xlim(-(outer_lik + 0.05), (outer_lik + 0.05))
        if density:
            plt.ylabel("Density")
        else:
            plt.ylabel("Frequency")
        plt.xlabel("10logLR")
        plt.legend(loc=9)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def plot_boxplot_of_metric(n_metric, name_metric, savefig=None, show=None):

    MLP_bin_soft, MLR_bin_soft, XGB_bin_soft = n_metric[:, 0, 0, :].T
    MLP_norm_soft, MLR_norm_soft, XGB_norm_soft = n_metric[:, 1, 0, :].T
    MLP_bin_sig, MLR_bin_sig, XGB_bin_sig = n_metric[:, 0, 1, :].T
    MLP_norm_sig, MLR_norm_sig, XGB_norm_sig = n_metric[:, 1, 1, :].T

    data = [MLP_bin_soft, MLR_bin_soft, XGB_bin_soft,
            MLP_norm_soft, MLR_norm_soft, XGB_norm_soft,
            MLP_bin_sig, MLR_bin_sig, XGB_bin_sig,
            MLP_norm_sig, MLR_norm_sig, XGB_norm_sig]

    names = ['MLP_bin_soft', 'MLR_bin_soft', 'XGB_bin_soft',
            'MLP_norm_soft', 'MLR_norm_soft', 'XGB_norm_soft',
            'MLP_bin_sig', 'MLR_bin_sig', 'XGB_bin_sig',
            'MLP_norm_sig', 'MLR_norm_sig', 'XGB_norm_sig']

    fig, ax = plt.subplots()
    ax.set_title("Boxplots of {} for {} folds".format(name_metric, n_metric.shape[0]))
    ax.boxplot(data, vert=False)
    ax.set_xlabel(name_metric)
    plt.yticks(list(range(1, len(names)+1)), names)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()

    plt.close(fig)


# TODO: Make function work
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
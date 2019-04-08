"""
Plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve

from rna.analytics import combine_samples
from rna.lr_system import convert_prob_per_mixture_to_marginal_per_class
from lir import PavLogLR


def boxplot_per_single_class_category(y_prob_per_class,
                                      y_augmented_matrix,
                                      classes_to_evaluate,
                                      classes_combinations_to_evaluate):
    """
    for single cell type, plot the distribution of marginal LRs for each cell type,
    as well as for specified combinations of classes.

    :param X_augmented_test: N_samples x N_markers array of observations
    :param y_augmented_matrix: N_samples x (N_single_cell_types + N_combos)
        n_hot encoding
    :param classes_to_evaluate: list of str, names of classes to evaluate
    :param mixtures_in_classes_of_interest: list of lists, specifying for each
        class in classes_to_evaluate which
    mixture labels contain these
    :param class_combinations_to_evaluate: list of lists of int, specifying
        combinations of single cell types to consider
    :return: None
    """

    classes_only_single = classes_to_evaluate.copy()
    for celltype, i_celltype in classes_to_evaluate.items():
        if 'and/or' in celltype:
            del classes_only_single[celltype]

    n_single_classes_to_draw = y_augmented_matrix.shape[1]
    # y_prob = model.predict_proba(X_augmented_test)
    # y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
    #     y_prob, mixtures_in_classes_of_interest, classes_map_updated, MAX_LR)
    log_lrs_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
    plt.subplots(2, int(n_single_classes_to_draw/2), figsize=(18, 9))
    for idx, (celltype, i_celltype) in enumerate(sorted(classes_to_evaluate.items())):
    # for i, celltype in enumerate(classes_to_evaluate):
        i_celltype = classes_to_evaluate[celltype]
        indices = [j for j in range(y_augmented_matrix.shape[0]) if
                   y_augmented_matrix[j, i_celltype] == 1
                   and sum(y_augmented_matrix[j, :]) == 1]
        plt.subplot(2, int(n_single_classes_to_draw/2), idx + 1)
        plt.xlim([-MAX_LR -.5, MAX_LR+.5])
        bplot = plt.boxplot(log_lrs_per_class[indices, :], vert=False,
                            labels=classes_to_evaluate, patch_artist=True)
        colors = ['white'] * (n_single_classes_to_draw + 1)
        colors[i_celltype] = 'black'
        for j, comb in enumerate(classes_combinations_to_evaluate):
            if celltype in comb:
                colors[n_single_classes_to_draw + j] = 'black'
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title(celltype)
    plt.show()

    #     for j, comb in enumerate(class_combinations_to_evaluate):
    #         if inv_classes_map[i] in comb:
    #             colors[n_single_classes_to_draw + j] = 'black'
    #     for patch, color in zip(bplot['boxes'], colors):
    #         patch.set_facecolor(color)
    #
    #     plt.title(inv_classes_map[i])
    # plt.savefig('singles boxplot')


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


# def plot_histogram_log_lr(h0_h1_lrs, n_bins=30, title='before', density=None, savefig=None, show=None):
#
#     celltypes = list(h0_h1_lrs.keys())
#     plt.subplots(int(len(celltypes)/2), 2, figsize=(9, 9/4*len(celltypes)), sharey='row')
#     # plt.suptitle('Histogram of log LRs {} calibration'.format(title), size=16)
#
#     minimum_lik = np.min([min(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
#     maximum_lik = np.max([max(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
#     outer_lik = max(abs(minimum_lik), abs(maximum_lik))
#
#     for idx, celltype in enumerate(sorted(celltypes)):
#         log_likrats0 = h0_h1_lrs[celltype][0]
#         log_likrats1 = h0_h1_lrs[celltype][1]
#
#         plt.subplot(int(len(celltypes)/2), 2, idx + 1)
#         plt.hist(log_likrats0, density=density, color='orange', label='h0', bins=n_bins, alpha=0.5)
#         plt.hist(log_likrats1, density=density, color='blue', label='h1', bins=n_bins, alpha=0.5)
#         plt.legend(loc='upper right')
#         if title == 'after':
#             plt.xlim(-(outer_lik + 0.05), (outer_lik + 0.05))
#         plt.ylabel("Density")
#         plt.xlabel("10logLR")
#         plt.title(celltype, fontsize=16)
#         # plt.text(0.2, 0.9, 'N_train = 100,\nN_test = 50,\nN_calibration = 4',
#         #          ha='center', va='center', transform=ax.transAxes)
#
#     if savefig is not None:
#         plt.tight_layout()
#         plt.savefig(savefig)
#     if show or savefig is None:
#         plt.show()


def plot_histogram_log_lr(lrs, y_nhot, target_classes, n_bins=30, title='before',
                          density=None, savefig=None, show=None):

    n_target_classes = len(target_classes)
    inv_y_nhot = 1-y_nhot
    plt.subplots(int(n_target_classes / 2), 2, figsize=(9, int(9 / 4 * n_target_classes)), sharey='row')
    for i, target_class in enumerate(target_classes):
        lrs1 = np.multiply(lrs[:, i], np.max(np.multiply(y_nhot, target_class), axis=1))
        lrs2 = np.multiply(lrs[:, i], np.max(np.multiply(inv_y_nhot, target_class), axis=1))

        plt.subplot(int(n_target_classes / 2), 2, i + 1)
        plt.hist(lrs1, density=density, color='orange', label='h1', bins=n_bins, alpha=0.5)
        plt.hist(lrs2, density=density, color='blue', label='h2', bins=n_bins, alpha=0.5)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


    # celltypes = list(h0_h1_lrs.keys())

    # plt.suptitle('Histogram of log LRs {} calibration'.format(title), size=16)

    # minimum_lik = np.min([min(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
    # maximum_lik = np.max([max(np.append(h0_h1_lrs[celltype][0], h0_h1_lrs[celltype][1])) for celltype in celltypes])
    # outer_lik = max(abs(minimum_lik), abs(maximum_lik))

    # for idx, celltype in enumerate(sorted(celltypes)):
    #     log_likrats0 = h0_h1_lrs[celltype][0]
    #     log_likrats1 = h0_h1_lrs[celltype][1]

        # plt.subplot(int(len(celltypes)/2), 2, idx + 1)
        # plt.hist(log_likrats0, density=density, color='orange', label='h0', bins=n_bins, alpha=0.5)
        # plt.hist(log_likrats1, density=density, color='blue', label='h1', bins=n_bins, alpha=0.5)
        # plt.legend(loc='upper right')
        # if title == 'after':
        #     plt.xlim(-(outer_lik + 0.05), (outer_lik + 0.05))
        # plt.ylabel("Density")
        # plt.xlabel("10logLR")
        # plt.title(celltype, fontsize=16)
        # plt.text(0.2, 0.9, 'N_train = 100,\nN_test = 50,\nN_calibration = 4',
        #          ha='center', va='center', transform=ax.transAxes)



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
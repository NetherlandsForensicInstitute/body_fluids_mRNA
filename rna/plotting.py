"""
Plotting functions.
"""

import numpy as np

from matplotlib import rc, pyplot as plt, patches as mpatches
from collections import OrderedDict

# from rna.analytics import combine_samples

from rna.utils import vec2string, prior2string

from lir import PavLogLR


rc('text', usetex=True)


def plot_calibration_process(lrs, y_nhot, calibrators, target_classes, label_encoder, calibration_on_loglrs, savefig=None,
                             show=None):

    for i, target_class in enumerate(target_classes):
        lr = lrs[:, i]
        calibrator = calibrators[str(target_class)]
        plot_calibration_process_per_target_class(lr, y_nhot, calibrator, target_class, label_encoder,
                                              calibration_on_loglrs)

        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save)
        if show or savefig is None:
            plt.show()

        plt.close()


def plot_calibration_process_per_target_class(lr, y_nhot, calibrator, target_class, label_encoder,
                                              calibration_on_loglrs):
    if calibration_on_loglrs:
        data = np.log10(lr)
        xlabel = '10logLR'
        min_val = -11
        max_val = 11
    else:
        data = lr / (1 + lr)
        xlabel = 'Probability'
        min_val = -0.1
        max_val = 1.1

    data1 = data[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1)]
    data2 = data[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0)]

    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 12))
    celltype = vec2string(target_class, label_encoder)
    plt.suptitle(celltype, y=1.05)

    # 1 histogram log10LRs without calibration
    axes[0, 0].hist(data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
    axes[0, 0].hist(data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_xlim(min_val, max_val)
    axes[0, 0].legend()

    # 2 histogram log10LRs without calibration + KDE curves
    LRs = np.ravel(sorted(data))
    calibrator.transform(LRs)
    axes[0, 1].hist(data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
    axes[0, 1].hist(data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
    axes[0, 1].plot(LRs, calibrator.p1, color='orange', label='p1')
    axes[0, 1].plot(LRs, calibrator.p0, color='blue', label='p0')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlim(min_val, max_val)

    # 3 KDE curves
    axes[1, 0].plot(LRs, calibrator.p1, color='orange', label='p1')
    axes[1, 0].plot(LRs, calibrator.p0, color='blue', label='p0')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_xlim(min_val, max_val)
    axes[1, 0].legend()

    # 4 Ratio of two curves
    ratio = calibrator.p1 / calibrator.p0
    if calibration_on_loglrs:
        X_abovemin10 = np.unique(np.linspace(min_val, min(LRs), 200))
        calibrator.transform(X_abovemin10)
        ratio_abovemin10 = calibrator.p1 / calibrator.p0

        X_below10 = np.unique(np.linspace(max(LRs), max_val, 200))
        calibrator.transform(X_below10)
        ratio_below10 = calibrator.p1 / calibrator.p0

    axes[1, 1].set_xlim(min_val, max_val)
    axes[1, 1].plot(LRs, ratio, color='green', label='ratio')
    if calibration_on_loglrs:
        axes[1, 1].plot(X_abovemin10, ratio_abovemin10, color='green', linestyle=':', linewidth=1)
        axes[1, 1].plot(X_below10, ratio_below10, color='green', linestyle=':', linewidth=1)
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel('Ratio p1/p0')

    # 5
    logratio = np.log10(ratio)
    axes[2, 0].plot(LRs, logratio, color='green', label='ratio')
    if calibration_on_loglrs:
        axes[2, 0].plot(X_abovemin10, np.log10(ratio_abovemin10), color='green', linestyle=':', linewidth=1)
        axes[2, 0].plot(X_below10, np.log10(ratio_below10), color='green', linestyle=':', linewidth=1)
    axes[2, 0].set_xlabel(xlabel)
    axes[2, 0].set_ylabel('10log Ratio p1/p0')
    axes[2, 0].set_xlim(min_val, max_val)

    # 6
    # axes[2, 1].hist(logratio, color='green', density=True, bins=30)
    # axes[2, 1].set_xlabel('10log Ratio p1/p0')
    # axes[2, 1].set_ylabel('Density')
    # axes[2, 1].set_xlim(-10.25, 10.25)

    # 7
    LRs = calibrator.transform(data)

    if calibration_on_loglrs:
        calibrated_data = np.log10(LRs)
        xlabel = 'Calibrated 10logLR'
    else:
        calibrated_data = LRs / (1 + LRs)
        xlabel = 'Calibrated probability'

    calibrated_data1 = calibrated_data[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1)]
    calibrated_data2 = calibrated_data[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0)]

    calibrated_data1 = np.reshape(calibrated_data1, -1)
    calibrated_data2 = np.reshape(calibrated_data2, -1)

    axes[3, 0].hist(calibrated_data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
    axes[3, 0].hist(calibrated_data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
    axes[3, 0].set_xlabel(xlabel)
    axes[3, 0].set_ylabel('Density')
    axes[3, 0].set_xlim(min_val, max_val)

    # 8
    if calibration_on_loglrs:
        axes[3, 1].hist(calibrated_data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
        axes[3, 1].hist(calibrated_data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
        axes[3, 1].set_xlabel(xlabel)
        axes[3, 1].set_ylabel('Density')
        axes[3, 1].set_xlim((np.min(calibrated_data) - 0.25), (np.max(calibrated_data) + 0.25))
    else:
        calibrated_data1 = np.log10(calibrated_data1 / (1 - calibrated_data1))
        calibrated_data2 = np.log10(calibrated_data2 / (1 - calibrated_data2))

        axes[3, 1].hist(calibrated_data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
        axes[3, 1].hist(calibrated_data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
        axes[3, 1].set_xlabel('Calibrated 10logLR')
        axes[3, 1].set_ylabel('Density')


def plot_scatterplot_lr_before_after_calib(lrs_before, lrs_after, y_nhot, target_classes, label_encoder, show=None,
                                           savefig=None):

    loglrs_before = np.log10(lrs_before)
    loglrs_after = np.log10(lrs_after)

    n_target_classes = len(target_classes)

    if n_target_classes > 1:
        n_rows = int(n_target_classes / 2)
        fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=False)

        j = 0
        k = 0

    for i, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        min_vals = [min(loglrs_before[:, i]), min(loglrs_after[:, i])]
        max_vals = [max(loglrs_before[:, i]), max(loglrs_after[:, i])]
        diagonal_coordinates = np.linspace(min(min_vals), max(max_vals))

        target_class = np.reshape(target_class, -1, 1)
        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        colors = ['orange' if l == 1.0 else 'blue' for l in labels]

        h1 = mpatches.Patch(color='orange', label='h1')
        h2 = mpatches.Patch(color='blue', label='h2')

        if n_target_classes == 1:

            plt.scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
            plt.plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
            plt.title(celltype)
            plt.xlim(min(min_vals), max(max_vals))
            plt.ylim(min(min_vals), max(max_vals))
            plt.legend(handles=[h1, h2])

            plt.xlabel("10logLRs before")
            plt.ylabel("10logLRs after")

        elif n_target_classes == 2:
            axs[i].scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
            axs[i].plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
            axs[i].set_title(celltype)
            axs[i].set_xlim(min(min_vals), max(max_vals))
            axs[i].set_ylim(min(min_vals), max(max_vals))

            fig.text(0.5, 0.04, "lrs before", ha='center')
            fig.text(0.04, 0.5, "lrs after", va='center', rotation='vertical')

        elif n_target_classes > 2:
            axs[j, k].scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
            axs[j, k].plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
            axs[j, k].set_title(celltype)
            axs[j, k].set_xlim(min(min_vals), max(max_vals))
            axs[j, k].set_ylim(min(min_vals), max(max_vals))

            if (i % 2) == 0:
                k = 1
            else:
                k = 0
                j = j + 1

            fig.text(0.5, 0.04, "lrs before", ha='center')
            fig.text(0.04, 0.5, "lrs after", va='center', rotation='vertical')

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

    plt.close()


def plot_histograms_all_lrs_all_folds(lrs_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                      show=None, savefig=None):

    for method in lrs_for_all_methods.keys():

        plot_histogram_lr_all_folds(lrs_for_all_methods[method], y_nhot_for_all_methods[method], target_classes, label_encoder)

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + method)
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()


def plot_histogram_lr_all_folds(lrs, y_nhot, target_classes, label_encoder, n_bins=30, title='after', density=True):

    loglrs = np.log10(lrs)
    n_target_classes = len(target_classes)

    if n_target_classes > 1:
        n_rows = int(n_target_classes / 2)
        if title == 'after':
            fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=False)
        else:
            fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=True)
        plt.suptitle('Histograms {} calibration'.format(title))

        j = 0
        k = 0

    for i, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        loglrs1 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1), i]
        loglrs2 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0), i]

        if n_target_classes == 1:
            plt.hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            plt.hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            plt.title(celltype)
            plt.legend()

        elif n_target_classes == 2:
            axs[i].hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            axs[i].hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            axs[i].set_title(celltype)

            handles, labels = axs[0].get_legend_handles_labels()

            fig.text(0.5, 0.04, "10logLR", ha='center')
            if density:
                fig.text(0.04, 0.5, "Density", va='center', rotation='vertical')
            else:
                fig.text(0.04, 0.5, "Frequency", va='center', rotation='vertical')

            fig.legend(handles, labels, 'center right')

        elif n_target_classes > 2:
            axs[j, k].hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            axs[j, k].hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            axs[j, k].set_title(celltype)

            if (i % 2) == 0:
                k = 1
            else:
                k = 0
                j = j + 1

            handles, labels = axs[0, 0].get_legend_handles_labels()

            fig.text(0.5, 0.04, "10logLR", ha='center')
            if density:
                fig.text(0.04, 0.5, "Density", va='center', rotation='vertical')
            else:
                fig.text(0.04, 0.5, "Frequency", va='center', rotation='vertical')

            fig.legend(handles, labels, 'center right')



def plot_scatterplots_all_lrs_different_priors(lrs_for_all_methods, y_nhot_for_all_methods, target_classes,
                                               label_encoder, show=None, savefig=None):

    methods_no_prior = []
    for method in lrs_for_all_methods.keys():
        methods_no_prior.append(method.split('[')[0])
    methods_no_prior = np.unique(methods_no_prior).tolist()

    test_dict = OrderedDict()
    for method in methods_no_prior:
        for names in lrs_for_all_methods.keys():
            if method in names:
                if method in test_dict:
                    test_dict[method].update({names: (lrs_for_all_methods[names], y_nhot_for_all_methods[names])})
                else:
                    test_dict[method] = {names: (lrs_for_all_methods[names], y_nhot_for_all_methods[names])}

    for keys, values in test_dict.items():
        for t, target_class in enumerate(target_classes):
            fig, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
            plt.suptitle(vec2string(target_class, label_encoder))
            plot_scatterplot_lrs_different_priors(values, t, target_class, label_encoder, ax=axs1, title='augmented test')

            target_class_str = vec2string(target_class, label_encoder)
            target_class_save = target_class_str.replace(" ", "_")
            target_class_save = target_class_save.replace(".", "_")
            target_class_save = target_class_save.replace("/", "_")
            if savefig is not None:
                plt.tight_layout()
                plt.savefig(savefig + '_' + keys + '_' + target_class_save)
                plt.close()
            if show or savefig is None:
                plt.tight_layout()
                plt.show()


def plot_scatterplot_lrs_different_priors(values, t, target_class, label_encoder, ax=None, title=None):

    ax = ax
    min_vals = []
    max_vals = []
    loglrs = OrderedDict()
    y_nhot = OrderedDict()
    full_name = []
    priors = []
    for method, data in values.items():
        loglrs[method] = np.log10(data[0][:, t])
        y_nhot[method] = data[1]
        full_name.append(method)
        priors.append('[' + method.split('[')[1])
        min_vals.append(np.min(np.log10(data[0][:, t])))
        max_vals.append(np.max(np.log10(data[0][:, t])))
    assert np.array_equal(y_nhot[full_name[0]], y_nhot[full_name[1]])
    diagonal_coordinates = np.linspace(min(min_vals), max(max_vals))

    target_class = np.reshape(target_class, -1, 1)
    labels = np.max(np.multiply(y_nhot[full_name[0]], target_class), axis=1)

    colors = ['orange' if l == 1.0 else 'blue' for l in labels]

    # make sure uniform priors always on bottom
    if any(str([1] * len(target_class)) in x for x in priors):
        index1 = priors.index(str([1] * len(target_class)))
        loglrs1 = loglrs[full_name[index1]]
        loglrs2 = loglrs[full_name[1-index1]]
    else:
        index1 = 0
        index2 = 1
        loglrs1 = loglrs[full_name[index1]]
        loglrs2 = loglrs[full_name[index2]]

    ax.scatter(loglrs1, loglrs2, s=3, color=colors, alpha=0.5)
    ax.plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
    ax.set_title(title)
    ax.set_xlim(min(min_vals), max(max_vals))
    ax.set_ylim(min(min_vals), max(max_vals))

    ax.set_xlabel("10logLR {}".format(prior2string(priors[index1], label_encoder)))
    ax.set_ylabel("10logLR {}".format(prior2string(priors[1 - index1], label_encoder)))

    return ax


# TODO: Make function generic
def plot_boxplot_of_metric(n_metric, name_metric, savefig=None, show=None):

    MLP_bin_soft_priorunif, MLR_bin_soft_priorunif, XGB_bin_soft_priorunif, DL_bin_soft_priorunif = n_metric[:, 0, 0, :, 0].T
    MLP_bin_soft_priorother, MLR_bin_soft_priorother, XGB_bin_soft_priorother, DL_bin_soft_priorother = n_metric[:, 0, 0, :, 1].T
    MLP_norm_soft_priorunif, MLR_norm_soft_priorunif, XGB_norm_soft_priorunif, DL_norm_soft_priorunif = n_metric[:, 1, 0, :, 0].T
    MLP_norm_soft_priorother, MLR_norm_soft_priorother, XGB_norm_soft_priorother, DL_norm_soft_priorother = n_metric[:, 1, 0, :, 1].T
    MLP_bin_sig_priorunif, MLR_bin_sig_priorunif, XGB_bin_sig_priorunif, DL_bin_sig_priorunif = n_metric[:, 0, 1, :, 0].T
    MLP_bin_sig_priorother, MLR_bin_sig_priorother, XGB_bin_sig_priorother, DL_bin_sig_priorother = n_metric[:, 0, 1, :, 1].T
    MLP_norm_sig_priorunif, MLR_norm_sig_priorunif, XGB_norm_sig_priorunif, DL_norm_sig_priorunif = n_metric[:, 1, 1, :, 0].T
    MLP_norm_sig_priorother, MLR_norm_sig_priorother, XGB_norm_sig_priorother, DL_norm_sig_priorother = n_metric[:, 1, 1, :, 0].T

    data = [MLP_bin_soft_priorunif, MLP_bin_soft_priorother, MLR_bin_soft_priorunif, MLR_bin_soft_priorother,
            XGB_bin_soft_priorunif, XGB_bin_soft_priorother, DL_bin_soft_priorunif, DL_bin_soft_priorother,
            MLP_norm_soft_priorunif, MLP_norm_soft_priorother, MLR_norm_soft_priorunif, MLR_norm_soft_priorother,
            XGB_norm_soft_priorunif, XGB_norm_soft_priorother, DL_norm_soft_priorunif, DL_norm_soft_priorother,
            MLP_bin_sig_priorunif, MLP_bin_sig_priorother, MLR_bin_sig_priorunif, MLR_bin_sig_priorother,
            XGB_bin_sig_priorunif, XGB_bin_sig_priorother, DL_bin_sig_priorunif, DL_bin_sig_priorother,
            MLP_norm_sig_priorunif, MLP_norm_sig_priorother, MLR_norm_sig_priorunif, MLR_norm_sig_priorother,
            XGB_norm_sig_priorunif, XGB_norm_sig_priorother, DL_norm_sig_priorunif, DL_norm_sig_priorother]

    names = ['MLP bin soft priorunif', 'MLP bin soft priorother', 'MLR bin soft priorunif', 'MLR bin soft priorother',
             'XGB bin soft priorunif', 'XGB bin soft priorother', 'DL bin soft priorunif', 'DL bin soft priorother',
             'MLP norm soft priorunif', 'MLP norm soft priorother', 'MLR norm soft priorunif', 'MLR norm soft priorother',
             'XGB norm soft priorunif', 'XGB norm soft priorother', 'DL norm soft priorunif', 'DL norm soft priorother',
             'MLP bin sig priorunif', 'MLP bin sig priorother', 'MLR bin sig priorunif', 'MLR bin sig priorother',
             'XGB bin sig priorunif', 'XGB bin sig priorother', 'DL bin sig priorunif', 'DL bin sig priorother',
             'MLP norm sig priorunif', 'MLP norm sig priorother', 'MLR norm sig priorunif', 'MLR norm sig priorother',
             'XGB norm sig priorunif', 'XGB norm sig priorother', 'DL norm sig priorunif', 'DL norm sig priorother']

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    fig.suptitle("{} folds".format(n_metric.shape[0]))
    axes[0].boxplot(data[0:8], vert=False)
    axes[0].set_yticklabels(names[0:8])

    axes[1].boxplot(data[8:16], vert=False)
    axes[1].set_yticklabels(names[8:16])

    axes[2].boxplot(data[16:24], vert=False)
    axes[2].set_yticklabels(names[16:24])

    axes[3].boxplot(data[24:32], vert=False)
    axes[3].set_yticklabels(names[24:32])
    axes[3].set_xlabel(name_metric)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


# TODO: Make this function work (?)
def plot_for_experimental_mixture_data(X_mixtures, y_mixtures, y_mixtures_matrix, inv_test_map, classes_to_evaluate,
                                       mixtures_in_classes_of_interest, n_single_cell_types_no_penile, dists):
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
    y_prob = model.predict_lrs(X_mixtures, )
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

# TODO: Make this funtion work?
# def plot_data(X):
#     """
#     plots the raw data points
#
#     :param X: N_samples x N_observations_per_sample x N_markers measurements
#     """
#     plt.matshow(combine_samples(X))
#     plt.savefig('single_cell_type_measurements_after_QC')


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


# TODO: Want to keep this?
# def plot_scatterplot_all_lrs_before_after_calib(lrs_before_for_all_methods, lrs_after_for_all_methods,
#                                                 y_nhot_for_all_methods, target_classes, label_encoder, show=None,
#                                                 savefig=None):
#     """
#     For each method plots the lrs
#
#     :param lrs_before_for_all_methods:
#     :param lrs_after_for_all_methods:
#     :param y_nhot_for_all_methods:
#     :param target_classes:
#     :param label_encoder:
#     :param show:
#     :param savefig:
#     :return:
#     """
#
#     for method in lrs_before_for_all_methods.keys():
#         plot_scatterplot_lr_before_after_calib(lrs_before_for_all_methods[method], lrs_after_for_all_methods[method],
#                                                y_nhot_for_all_methods[method], target_classes, label_encoder)
#
#         if savefig is not None:
#             plt.tight_layout()
#             plt.savefig(savefig + '_' + method)
#             plt.close()
#         if show or savefig is None:
#             plt.tight_layout()
#             plt.show()


# def plot_scatterplot_lr_before_after_calib(lrs_before, lrs_after, y_nhot, target_classes, label_encoder):
#
#     loglrs_before = np.log10(lrs_before)
#     loglrs_after = np.log10(lrs_after)
#
#     n_target_classes = len(target_classes)
#
#     if n_target_classes > 1:
#         n_rows = int(n_target_classes / 2)
#         fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=False)
#
#         j = 0
#         k = 0
#
#     for i, target_class in enumerate(target_classes):
#
#         celltype = vec2string(target_class, label_encoder)
#
#         min_vals = [min(loglrs_before), min(loglrs_after)]
#         max_vals = [max(loglrs_before), max(loglrs_after)]
#         diagonal_coordinates = np.linspace(min(min_vals), max(max_vals))
#
#         target_class = np.reshape(target_class, -1, 1)
#         labels = np.max(np.multiply(y_nhot, target_class), axis=1)
#
#         colors = ['orange' if l == 1.0 else 'blue' for l in labels]
#
#         if n_target_classes == 1:
#
#             plt.scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
#             plt.plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
#             plt.title(celltype)
#             plt.xlim(min(min_vals), max(max_vals))
#             plt.ylim(min(min_vals), max(max_vals))
#
#             plt.xlabel("lrs before")
#             plt.ylabel("lrs after")
#
#         elif n_target_classes == 2:
#             axs[i].scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
#             axs[i].plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
#             axs[i].set_title(celltype)
#             axs[i].set_xlim(min(min_vals), max(max_vals))
#             axs[i].set_ylim(min(min_vals), max(max_vals))
#
#             fig.text(0.5, 0.04, "lrs before", ha='center')
#             fig.text(0.04, 0.5, "lrs after", va='center', rotation='vertical')
#
#         elif n_target_classes > 2:
#             axs[j, k].scatter(loglrs_before[:, i], loglrs_after[:, i], s=3, color=colors, alpha=0.5)
#             axs[j, k].plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
#             axs[j, k].set_title(celltype)
#             axs[j, k].set_xlim(min(min_vals), max(max_vals))
#             axs[j, k].set_ylim(min(min_vals), max(max_vals))
#
#             if (i % 2) == 0:
#                 k = 1
#             else:
#                 k = 0
#                 j = j + 1
#
#             fig.text(0.5, 0.04, "lrs before", ha='center')
#             fig.text(0.04, 0.5, "lrs after", va='center', rotation='vertical')
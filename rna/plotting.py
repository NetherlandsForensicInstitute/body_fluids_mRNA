"""
Plotting functions.
"""

import math
import numpy as np
import rna.settings as settings

from matplotlib import rc, pyplot as plt, patches as mpatches
from matplotlib import colors as mcolors
# from matplotlib.cm import get_cmap
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# from rna.analytics import combine_samples

from rna.constants import celltype_specific_markers
from rna.utils import vec2string, prior2string, bool2str_binarize, bool2str_softmax
from rna.lr_system import get_mixture_columns_for_class

from lir import PavLogLR
from lir.calibration import IsotonicCalibrator


rc('text', usetex=True)


def plot_calibration_process(lrs, y_nhot, calibrators, true_lrs, target_classes, label_encoder, calibration_on_loglrs,
                             savefig=None, show=None):

    for t, target_class in enumerate(target_classes):
        lr = lrs[:, t]
        calibrator = calibrators[str(target_class)]
        if true_lrs is not None:
            plot_calibration_process_per_target_class(lr, y_nhot, calibrator, (true_lrs[0][:, t], true_lrs[1][:, t]), target_class, label_encoder,
                                                      calibration_on_loglrs)
        else:
            plot_calibration_process_per_target_class(lr, y_nhot, calibrator, None, target_class, label_encoder,
                                                      calibration_on_loglrs)

        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save)
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close()


def plot_calibration_process_per_target_class(lr, y_nhot, calibrator, true_lrs, target_class, label_encoder,
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
    axes[0, 0].legend(loc='upper right')

    # 2 histogram log10LRs without calibration + KDE curves
    LRs = np.ravel(sorted(data))
    calibrator.transform(LRs)
    axes[0, 1].hist(data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
    axes[0, 1].hist(data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
    axes[0, 1].plot(LRs, calibrator.p1, color='orange', label='p1')
    axes[0, 1].plot(LRs, calibrator.p0, color='blue', label='p2')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlim(min_val, max_val)

    # 3 KDE curves
    axes[1, 0].plot(LRs, calibrator.p1, color='orange', label='p1')
    axes[1, 0].plot(LRs, calibrator.p0, color='blue', label='p2')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_xlim(min_val, max_val)
    axes[1, 0].legend(loc='upper right')

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
    axes[1, 1].set_ylabel('Ratio p1/p2')

    # 5
    logratio = np.log10(ratio)
    axes[2, 0].plot(LRs, logratio, color='green', label='ratio')
    if calibration_on_loglrs:
        axes[2, 0].plot(X_abovemin10, np.log10(ratio_abovemin10), color='green', linestyle=':', linewidth=1)
        axes[2, 0].plot(X_below10, np.log10(ratio_below10), color='green', linestyle=':', linewidth=1)
    axes[2, 0].set_xlabel(xlabel)
    axes[2, 0].set_ylabel('10log Ratio p1/p2')
    axes[2, 0].set_xlim(min_val, max_val)

    # 6
    if true_lrs is not None:
        lrs_before, lrs_after = (true_lrs)
        loglrs_before = np.log10(lrs_before)
        loglrs_after = np.log10(lrs_after)

        min_vals = [min(loglrs_before), min(loglrs_after)]
        max_vals = [max(loglrs_before), max(loglrs_after)]
        diagonal_coordinates = np.linspace(min(min_vals), max(max_vals))

        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        colors = ['orange' if l == 1.0 else 'blue' for l in labels]

        h1 = mpatches.Patch(color='orange', label='h1')
        h2 = mpatches.Patch(color='blue', label='h2')

        axes[2, 1].scatter(loglrs_before, loglrs_after, s=3, color=colors, alpha=0.2)
        axes[2, 1].plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
        axes[2, 1].set_xlim(min(min_vals), max(max_vals))
        axes[2, 1].set_ylim(min(min_vals), max(max_vals))
        axes[2, 1].set_xlabel("True 10logLRs before")
        axes[2, 1].set_ylabel("True 10logLRs after")
        axes[2, 1].legend(handles=[h1, h2], loc='upper right')

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

        plt.close()


def plot_histogram_lr_all_folds(lrs, y_nhot, target_classes, label_encoder, n_bins=30, title='after', density=True):

    loglrs = np.log10(lrs)
    n_target_classes = len(target_classes)

    if n_target_classes > 1:
        n_rows = math.ceil(n_target_classes / 2)
        if title == 'after':
            fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=False)
        else:
            fig, axs = plt.subplots(n_rows, 2, figsize=(9, int(9 / 4 * n_target_classes)), sharex=True, sharey=True)
        # plt.suptitle('Histograms {} calibration'.format(title))

        j = 0
        k = 0

    for t, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        loglrs1 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1), t]
        loglrs2 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0), t]

        if n_target_classes == 1:
            plt.hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            plt.hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            plt.title(celltype)
            plt.xlabel("10logLR")
            if density:
                plt.ylabel("Density")
            else:
                plt.ylabel("Frequency")
            plt.legend(loc='upper right')

        elif n_rows == 1:
            axs[t].hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            axs[t].hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            axs[t].set_title(celltype)

            handles, labels = axs[0].get_legend_handles_labels()

            fig.text(0.5, 0.002, "10logLR", ha='center')
            if density:
                fig.text(0.002, 0.5, "Density", va='center', rotation='vertical')
            else:
                fig.text(0.002, 0.5, "Frequency", va='center', rotation='vertical')

            fig.legend(handles, labels, 'center right')

        elif n_rows > 1:
                axs[j, k].hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
                axs[j, k].hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
                axs[j, k].set_title(celltype)

                if (t % 2) == 0:
                    k = 1
                else:
                    k = 0
                    j = j + 1

                handles, labels = axs[0, 0].get_legend_handles_labels()

                fig.text(0.5, 0.002, "10logLR", ha='center')
                if density:
                    fig.text(0.002, 0.5, "Density", va='center', rotation='vertical')
                else:
                    fig.text(0.002, 0.5, "Frequency", va='center', rotation='vertical')

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
            fig, (axs1, axs2, axs3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

            loglrs = OrderedDict()
            y_nhot = OrderedDict()
            full_name = []
            priors = []
            for method, data in values.items():
                loglrs[method] = np.log10(data[0][:, t])
                y_nhot[method] = data[1]
                full_name.append(method)
                priors.append('[' + method.split('[')[1])
            assert np.array_equal(y_nhot[full_name[0]], y_nhot[full_name[1]])

            target_class = np.reshape(target_class, -1, 1)
            labels = np.max(np.multiply(y_nhot[full_name[0]], target_class), axis=1)

            colors = ['orange' if l == 1.0 else 'blue' for l in labels]
            colors_positive = ['orange'] * int(np.sum(labels))
            colors_negative = ['blue'] * int((len(labels) - np.sum(labels)))

            h1 = mpatches.Patch(color='orange', label='h1')
            h2 = mpatches.Patch(color='blue', label='h2')

            # make sure uniform priors always on bottom
            if any(str([1] * len(target_class)) in x for x in priors):
                index1 = priors.index(str([1] * len(target_class)))
                loglrs1 = loglrs[full_name[index1]]
                loglrs2 = loglrs[full_name[1 - index1]]
            else:
                index1 = 0
                index2 = 1
                loglrs1 = loglrs[full_name[index1]]
                loglrs2 = loglrs[full_name[index2]]

            loglrs1_pos = loglrs1[np.argwhere(labels == 1)]
            loglrs2_pos = loglrs2[np.argwhere(labels == 1)]
            loglrs1_neg = loglrs1[np.argwhere(labels == 0)]
            loglrs2_neg = loglrs2[np.argwhere(labels == 0)]

            min_val_pos = math.floor(min(np.min(loglrs1_pos), np.min(loglrs2_pos)))
            max_val_pos = math.ceil(max(np.max(loglrs1_pos), np.max(loglrs2_pos)))
            min_val_neg = math.floor(min(np.min(loglrs1_neg), np.min(loglrs2_neg)))
            max_val_neg = math.ceil(max(np.max(loglrs1_neg), np.max(loglrs2_neg)))

            plot_scatterplot_lrs_different_priors((loglrs1, loglrs2), np.linspace(-4, 11), colors, (h1, h2), ax=axs1)
            plot_scatterplot_lrs_different_priors((loglrs1_pos, loglrs2_pos), np.linspace(min_val_pos, max_val_pos), colors_positive,
                                                  (h1), ax=axs2)
            plot_scatterplot_lrs_different_priors((loglrs1_neg, loglrs2_neg), np.linspace(min_val_neg, max_val_neg), colors_negative,
                                                  (h2), ax=axs3)

            fig.text(0.5, 0.002, "10logLR {}".format(prior2string(priors[index1], label_encoder)), ha='center')
            fig.text(0.002, 0.5, "10logLR {}".format(prior2string(priors[1 - index1], label_encoder)), va='center', rotation='vertical')

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


def plot_scatterplot_lrs_different_priors(loglrs, diagonal_coordinates, colors, handles, ax=None):

    ax = ax

    rect_neg = mpatches.Rectangle((-5, -5), 5, 5, color='blue', alpha=0.1, linewidth=0)
    rect_pos1 = mpatches.Rectangle((-5, 0), 16, 11, color='orange', alpha=0.1, linewidth=0)
    rect_pos2 = mpatches.Rectangle((0, -5), 11, 5, color='orange', alpha=0.1, linewidth=0)

    ax.plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
    ax.scatter(loglrs[0], loglrs[1], s=3, color=colors, alpha=0.2)
    ax.add_patch(rect_neg)
    ax.add_patch(rect_pos1)
    ax.add_patch(rect_pos2)
    ax.set_xlim(min(diagonal_coordinates), max(diagonal_coordinates))
    ax.set_ylim(min(diagonal_coordinates), max(diagonal_coordinates))

    try:
        ax.legend(handles=[handles[0], handles[1]], loc=0)
    except TypeError:
        ax.legend(handles=[handles], loc=0)

    return ax


def plot_boxplot_of_metric(n_metric, label_encoder, target_class, name_metric, savefig=None, show=None):

    def int2string_models(int, specify_which=None):
        if specify_which == None:
            raise ValueError("type must be set: 1 = transformation, 2 = probscalculations, 3 = model, 4 = prior")
        elif specify_which == 1:
            for i in range(len(settings.binarize)):
                if int == i:
                    return bool2str_binarize(settings.binarize[i])
        elif specify_which == 2:
            for i in range(len(settings.softmax)):
                if int == i:
                    return bool2str_softmax(settings.softmax[i])
        elif specify_which == 3:
            for i in range(len(settings.models)):
                if int == i:
                    return settings.models[i][0]
        elif specify_which == 4:
            for i in range(len(settings.priors)):
                if int == i:
                    return prior2string(str(settings.priors[i]), label_encoder)
        else:
            raise ValueError("Value {} for variable 'specify which' does not exist".format(specify_which))

    i_transformations = n_metric.shape[1]
    j_probscalulations = n_metric.shape[2]
    k_models = n_metric.shape[3]
    p_priors = n_metric.shape[4]

    total_boxplots = i_transformations * j_probscalulations * p_priors * k_models

    fig = plt.figure(figsize=(14, 7))
    # plt.suptitle(vec2string(target_class, label_encoder))
    ax = plt.subplot(111)
    ax.set_axisbelow(True)
    a = 0
    names = []
    for i in range(i_transformations):
        for j in range(j_probscalulations):
            for k in range(k_models):
                for p in range(p_priors):
                    names.append(int2string_models(k, 3) + ' ' + int2string_models(i, 1) + ' ' + int2string_models(j, 2) + ' ' + int2string_models(p, 4))
                    ax.boxplot(n_metric[:, i, j, k, p], vert=False, positions=[a], widths = 0.6)
                    a += 1
    ax.set_yticks(range(total_boxplots))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.5, total_boxplots-0.5)
    ax.set_xlabel(name_metric)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def plot_progress_of_metric(n_metric, label_encoder, target_class, name_metric, savefig=None, show=None):

    def int2string_models(int, specify_which=None):
        if specify_which == None:
            raise ValueError("type must be set: 1 = transformation, 2 = probscalculations, 3 = model, 4 = prior")
        elif specify_which == 1:
            for i in range(len(settings.binarize)):
                if int == i:
                    return bool2str_binarize(settings.binarize[i])
        elif specify_which == 2:
            for i in range(len(settings.softmax)):
                if int == i:
                    return bool2str_softmax(settings.softmax[i])
        elif specify_which == 3:
            for i in range(len(settings.models)):
                if int == i:
                    return settings.models[i][0]
        elif specify_which == 4:
            for i in range(len(settings.priors)):
                if int == i:
                    return prior2string(str(settings.priors[i]), label_encoder)
        else:
            raise ValueError("Value {} for variable 'specify which' does not exist".format(specify_which))

    n_folds = n_metric.shape[0]

    i_transformations = n_metric.shape[1]
    j_probscalulations = n_metric.shape[2]
    k_models = n_metric.shape[3]
    p_priors = n_metric.shape[4]

    fig = plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)
    x_lim = np.linspace(1, n_folds, n_folds)
    min_vals = []
    max_vals = []
    a = 0
    colors = cycle(('lightcoral', 'orangered', 'chocolate', 'orange', 'goldenrod', 'yellow', 'greenyellow',
                    'darkolivegreen', 'springgreen', 'turquoise', 'teal', 'darkturquoise',
                    'deepskyblue', 'steelblue', 'cornflowerblue', 'midnightblue', 'blue', 'slateblue', 'blueviolet',
                    'violet', 'magenta', 'deeppink', 'crimson'))
    markers = cycle(('+', 'o', 'x', '.', 'v', 'p', 's', 'P', '*', 'h', 'X', 'd', 'D'))

    for i in range(i_transformations):
        for j in range(j_probscalulations):
            for k in range(k_models):
                for p in range(p_priors):
                    name = int2string_models(k, 3) + ' ' + int2string_models(i, 1) + ' ' + int2string_models(j, 2) + ' ' + int2string_models(p, 4)
                    ax.plot(x_lim, n_metric[:, i, j, k, p], color=next(colors), label=name, marker=next(markers), linewidth=0.5, linestyle='--')
                    min_vals.append(np.min(n_metric[:, i, j, k, p]))
                    max_vals.append(np.max(n_metric[:, i, j, k, p]))
                    a += 1
    ax.set_xticks(x_lim)
    ax.set_xlabel("n fold")
    ax.set_ylabel(name_metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()
    plt.close(fig)

def plot_pavs(lrs_before, lrs_after, y_nhot, target_classes, label_encoder, savefig=None, show=None):

    for t, target_class in enumerate(target_classes):

        loglrs_before = np.log10(lrs_before[:, t])
        loglrs_after = np.log10(lrs_after[:, t])
        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        fig, (axs1, axs2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
        # plt.suptitle(vec2string(target_class, label_encoder))

        plot_pav(loglrs_before, labels, 'Before calibration', axs1)
        plot_pav(loglrs_after, labels, 'After calibration', axs2)

        fig.text(0.5, 0.001, 'pre-PAVcalibrated 10logLR', ha='center')
        fig.text(0.001, 0.5, 'post-PAVcalibrated 10logLR', va='center', rotation='vertical')

        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save)
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close(fig)


def plot_pav(loglr, labels, title, ax, show_scatter=True):
    """
    Plots pav plots for all cell types before and after calibration.
    :param title:
    """

    ax=ax

    pav = PavLogLR()
    pav_loglrs = pav.fit_transform(loglr, labels)
    xrange = [-10, 10]

    ax.axis('equal')
    ax.axis(xrange + xrange)
    ax.plot(xrange, xrange, color='black')

    pav_x = np.arange(*xrange, .01)
    ax.plot(pav_x, pav.transform(pav_x), color='green', alpha=0.9)
    ax.grid(True, linestyle=':')
    ax.set_title(title)
    if show_scatter:
        colors = ['orange' if l == 1.0 else 'blue' for l in labels]
        h1 = mpatches.Patch(color='orange', label='h1')
        h2 = mpatches.Patch(color='blue', label='h2')
        ax.scatter(loglr, pav_loglrs, color=colors)
        ax.legend(handles=[h1, h2], loc='upper right')
    ax.set_xticks(np.linspace(-10, 10, 8+1))
    ax.set_yticks(np.linspace(-10, 10, 8+1))

    return ax


def plot_insights_cllr(lrs_after, y_nhot, target_classes, label_encoder, savefig=None, show=None):

    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        labels = np.max(np.multiply(y_nhot, target_class), axis=1)
        plot_insight_cllr(lrs_after[:, t], labels, savefig=savefig + '_' + target_class_save, show=show)


def plot_insight_cllr(lrs, labels, savefig=None, show=None):
    def plot_ece(lrs, labels, ax):

        def pav_transform_lrs(lrs, labels):
            ir = IsotonicCalibrator()
            ir.fit(lrs, labels)
            lrs_after_calibration = ir.transform(lrs)

            lrs_after_calibration = np.where(lrs_after_calibration > 10 ** 10, 10 ** 10, lrs_after_calibration)
            lrs_after_calibration = np.where(lrs_after_calibration < 10 ** -10, 10 ** -10, lrs_after_calibration)

            lrs_after_calibration_p = lrs_after_calibration[np.argwhere(labels == 1)]
            lrs_after_calibration_d = lrs_after_calibration[np.argwhere(labels == 0)]

            return lrs_after_calibration_p, lrs_after_calibration_d

        def emperical_cross_entropy(lr_p, lr_d, prior):
            """

            :param lr_p: LRs with ground truth label prosecution
            :param prior_p: int; fixed value for prior prosecution
            :param lr_d: LRs with groudn trut label defence
            :param prior_d: int; fixed value for prior defence
            :return:
            """

            N_p = len(lr_p)
            N_d = len(lr_d)

            prior_p = prior
            prior_d = 1 - prior
            odds = prior_p / prior_d

            return (prior_p / N_p) * np.sum(np.log2(1 + (1 / (lr_p * odds)))) + \
                   (prior_d / N_d) * np.sum(np.log2(1 + (lr_d * odds))), odds

        def calculate_cllr(lr_p, lr_d):
            N_p = len(lr_p)
            N_d = len(lr_d)

            sum_p = np.sum(np.log2((1 + (1 / lr_p))))
            sum_d = np.sum(np.log2(1 + lr_d))

            return 0.5 * (((1 / N_p) * sum_p) + ((1 / N_d) * sum_d))

        ax = ax

        priors = np.linspace(0.001, 1 - 0.001, 50).tolist()

        lrs_p = lrs[np.argwhere(labels == 1)]
        lrs_d = lrs[np.argwhere(labels == 0)]

        # LR = 1
        results_LR_1 = np.array(
            [emperical_cross_entropy(np.ones_like(lrs_p), np.ones_like(lrs_d), prior) for prior in priors])
        ece_LR_1 = results_LR_1[:, 0]
        odds = results_LR_1[:, 1]

        # True LR
        results = np.array([emperical_cross_entropy(lrs_p, lrs_d, prior) for prior in priors])
        ece = results[:, 0]

        # LR after calibration
        lrs_after_calibration_p, lrs_after_calibration_d = pav_transform_lrs(lrs, labels)
        results_after_calib = np.array([emperical_cross_entropy(lrs_after_calibration_p, lrs_after_calibration_d, prior)
                                        for prior in priors])
        ece_after_calib = results_after_calib[:, 0]

        ax.plot(np.log10(odds), ece_LR_1, color='black', linestyle='--', label='LR=1 always (Cllr = {0:.1f})'.format(
            calculate_cllr(np.ones_like(lrs_p), np.ones_like(lrs_d))))
        ax.plot(np.log10(odds), ece, color='red',
                label='LR values (Cllr = {0:.3f})'.format(calculate_cllr(lrs_p, lrs_d)))
        ax.plot(np.log10(odds), ece_after_calib, color='darkgray', linestyle='-',
                label='LR after PAV (Cllr = {0:.3f})'.format(
                    calculate_cllr(lrs_after_calibration_p, lrs_after_calibration_d)))

        ax.set_ylabel("Emperical Cross-Entropy")
        ax.set_xlabel("Prior 10logOdds")
        ax.legend(loc='upper right')

        return ax

    def plot_punishment(ax, min_val, max_val):

        ax = ax

        sim_lrs = np.linspace(min_val, max_val, 1000000)

        sim_punish_p = np.log2(1 + (1 / sim_lrs))
        sim_punish_d = np.log2(1 + sim_lrs)

        ax.plot(np.log10(sim_lrs), sim_punish_p, color='orange')
                # label='punish p (max = {0:.1f})'.format(np.max(sim_punish_p)))
        ax.plot(np.log10(sim_lrs), sim_punish_d, color='blue')
                # label='punish d (max = {0:.1f})'.format(np.max(sim_punish_d)))

        ax.set_ylabel("Cost")
        ax.set_xlabel("10logLR")

        return ax

    def plot_true_punishment(lrs, labels, ax):

        ax = ax

        lrs_p = lrs[np.argwhere(labels == 1)]
        lrs_d = lrs[np.argwhere(labels == 0)]

        Np = len(lrs_p)
        Nd = len(lrs_d)

        punish_p = np.log2(1 + (1 / lrs_p))
        punish_d = np.log2(1 + lrs_d)

        ax.scatter(np.log10(lrs_p), punish_p, color='orange', marker='+', alpha=0.7,
                   label='h1: Cost = %1.1f, Relative cost = %2.2f' % (np.sum(punish_p), np.sum(punish_p) / Np))
        ax.scatter(np.log10(lrs_d), punish_d, color='blue', marker='x', alpha=0.7,
                   label='h2: Cost = %1.1f, Relative cost = %2.2f' % (np.sum(punish_d), np.sum(punish_d) / Nd))

        ax.set_title('Nd = {} and Np = {}'.format(len(lrs_p), len(lrs_d)))
        ax.set_ylabel("Cost")
        ax.set_xlabel('10logLR')
        ax.legend(loc='upper right')

        return ax

    loglrs_p = np.log10(lrs[np.argwhere(labels == 1)])
    loglrs_d = np.log10(lrs[np.argwhere(labels == 0)])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

    axes[0, 0].hist(loglrs_p, color='orange', label='h1', alpha=0.5, density=True)
    axes[0, 0].hist(loglrs_d, color='blue', label='h2', alpha=0.5, density=True)
    axes[0, 0].set_xlim(-5, 11)
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_xlabel('10logLR')
    axes[0, 0].legend(loc='upper right')

    plot_ece(lrs, labels, ax=axes[0, 1])
    plot_punishment(ax=axes[1, 0], min_val=np.min(lrs), max_val=np.max(lrs))
    plot_true_punishment(lrs, labels, ax=axes[1, 1])

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()


def plot_rocs(lrs_all_methods, y_nhot_all_methods, target_classes, label_encoder, savefig=None, show=None):

    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        plot_roc(lrs_all_methods, y_nhot_all_methods, t, target_class)

        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save)
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close()


def plot_roc(lrs_all_methods, y_nhot_all_methods, t, target_class):

    n_methods = len(lrs_all_methods.keys())
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    method_name = dict()
    for i, method in enumerate(lrs_all_methods.keys()):
        # TODO: On probs or lrs?
        lrs = np.log10(lrs_all_methods[method][:, t])
        # probs = lrs_all_methods[method][:, t] / (1 + lrs_all_methods[method][:, t])
        y_nhot = y_nhot_all_methods[method]
        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        fpr[i], tpr[i], _ = roc_curve(labels, lrs)
        roc_auc[i] = auc(fpr[i], tpr[i])

        method = method.replace("_", " ")
        method_name[i] = method

    lw=1.5
    colors=cycle(('lightcoral', 'orangered', 'chocolate', 'orange', 'goldenrod', 'yellow', 'greenyellow',
                    'darkolivegreen', 'springgreen', 'turquoise', 'teal', 'darkturquoise',
                    'deepskyblue', 'steelblue', 'cornflowerblue', 'midnightblue', 'blue', 'slateblue', 'blueviolet',
                    'violet', 'magenta', 'deeppink', 'crimson'))
    linestyle = cycle(('-', '-.', '--', ':', '-.'))

    plt.figure(figsize=(14, 7))
    for i in range(n_methods):
        plt.plot(fpr[i], tpr[i], color=next(colors), lw=lw, linestyle=next(linestyle),
                 label='method {0} (area = {1:0.2f})'''.format(method_name[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_coefficient_importances(model, target_classes, present_markers, label_encoder, savefig=None, show=None):

    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        celltype = target_class_str.split(' and/or ')
        # TODO: Is this correct for MLP with softmax?
        if len(model._classifier.coef_) == 2 ** 8:
            indices_target_class = get_mixture_columns_for_class(target_class, None)
            coefficients = np.mean(model._classifier.coef_[indices_target_class, :], axis=0)
        else:
            coefficients = model._classifier.coef_[t, :]

        plot_coefficient_importance(coefficients, present_markers, celltype)

        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save)
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close()


def plot_coefficient_importance(coefficients, present_markers, celltypes):
    """

    :param coefficients:
    :param present_markers:
    :param celltypes: list of strings: list of celltypes
    :return:
    """

    def calculate_maximum_lr(coefficients):
        positive_coefficients = coefficients[np.argwhere(coefficients > 0).ravel()]
        max_probability = 1 / (1 + np.exp(-(np.sum(positive_coefficients))))
        max_lr = max_probability / (1 - max_probability)
        if max_lr > 10 ** 10:
            return 10 ** 10
        else:
            return max_lr

    coefficients = np.reshape(coefficients, -1)
    max_lr = calculate_maximum_lr(coefficients)

    # sort
    sorted_indices = np.argsort(coefficients)
    coefficients = coefficients[sorted_indices]
    present_markers = np.array(present_markers)[sorted_indices].tolist()
    x = np.linspace(1, len(coefficients), len(coefficients))

    # get the indices of the celltype specific markers
    marker_indices = []
    for celltype in celltypes:
        for marker in celltype_specific_markers[celltype]:
            if marker is not None:
                marker_indices.append(present_markers.index(marker))
    marker_indices = np.unique(marker_indices)

    barlist = plt.barh(x, coefficients, color='grey', alpha=0.6, label='other')
    for marker_index in marker_indices:
        # highlight the markers that are celltype specific
        barlist[marker_index].set_color('navy')
        barlist[marker_index].set_hatch("/")
    try:
        barlist[marker_indices[0]].set_label('cell type specific')
    except IndexError:
        pass
    plt.yticks(x, present_markers)

    plt.title('Max lr = {}'.format(math.ceil(max_lr)))
    plt.xlabel('Coefficient value')
    plt.ylabel('Marker names')

    plt.legend(loc='lower right')


def plot_per_feature(model, augmented_data, target_classes, present_markers, train=True, savefig=None, show=None):

    # present_markers = ['HBB', 'ALAS2', 'CD93', 'HTN3', 'STATH', 'BPIFA1', 'MUC4', 'MYOZ1', 'CYP2B7P1', 'MMP10', 'MMP7',
    #                    'MMP11', 'SEMG1', 'KLK3', 'PRM1', 'RPS4Y1', 'XIST', 'ACTB', '18S-rRNA']
    present_markers = present_markers[:-4]
    augmented_data = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]']
    if train:
        X = augmented_data.X_train_augmented
        y_nhot = augmented_data.y_train_nhot_augmented
        title='Train'
    else:
        X = augmented_data.X_test_augmented
        y_nhot = augmented_data.y_test_nhot_augmented
        title='Test'

    for t, target_class in enumerate(target_classes):
        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        h1 = mpatches.Patch(color='orange', label='h1')
        h2 = mpatches.Patch(color='blue', label='h2')

        colors = ['orange' if l == 1.0 else 'blue' for l in labels]
        # colors_test = ['orange' if l == 1.0 else 'blue' for l in labels_test]

        # coefficients = model['[1, 1, 1, 1, 1, 1, 1, 1]']._classifier.coef_[t, :]

        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 9))
        plt.suptitle(title)
        j = 0
        k = 0
        for i, marker in enumerate(present_markers):
            # # xrange = np.linspace(min(X[:, i]), max(X[:, i]))
            # # xrange_test = np.linspace(min(X_test_augm_markers), max(X_test_augm_markers))
            #
            axes[j, k].scatter(X[:, i], labels, color=colors, s=1.5)
            # axes[j, k].set_xlim(np.linspace(min(X), max(X), 10))
            # # axes[i].plot(xrange_train, 1 / (1 + np.exp(-(coefficient * xrange_train))), color='darkgray', lw=2)
            axes[j, k].set_title(marker)
            axes[j, k].set_yticks([0, 1])

            k += 1
            if i == 3:
                k = 0
                j = 1
            elif i == 7:
                k = 0
                j = 2
            elif i == 11:
                k = 0
                j = 3

        plt.tight_layout()
        plt.show()

        # plt.legend(handles=[h1, h2], loc='center left', bbox_to_anchor=(1, 0.5))


# TODO: Not sure if correct
# def plot_feature_values_classes(model, augmented_data, target_classes, present_markers, savefig=None, show=None):
#
#     augmented_data = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]']
#     y_nhot_train = augmented_data.y_train_nhot_augmented
#     y_nhot_test = augmented_data.y_test_nhot_augmented
#
#     markers = ['CYP2B7P1', 'BPIFA1', 'MUC4', 'MYOZ1']
#     # present_markers = ['HBB', 'ALAS2', 'CD93', 'HTN3', 'STATH', 'BPIFA1', 'MUC4', 'MYOZ1', 'CYP2B7P1', 'MMP10', 'MMP7',
#     #                    'MMP11', 'SEMG1', 'KLK3', 'PRM1', 'RPS4Y1', 'XIST', 'ACTB', '18S-rRNA']
#     indices_markers = []
#     for marker in markers:
#         indices_markers.append(present_markers.index(marker))
#
#     for t, target_class in enumerate(target_classes):
#         labels_train = np.max(np.multiply(y_nhot_train, target_class), axis=1)
#         labels_test = np.max(np.multiply(y_nhot_test, target_class), axis=1)
#
#         h1 = mpatches.Patch(color='orange', label='h1')
#         h2 = mpatches.Patch(color='blue', label='h2')
#
#         colors_train = ['orange' if l == 1.0 else 'blue' for l in labels_train]
#         colors_test = ['orange' if l == 1.0 else 'blue' for l in labels_test]
#
#         coefficients = model['[1, 1, 1, 1, 1, 1, 1, 1]']._classifier.coef_[t, :]
#
#         fig_train, axes_train = plt.subplots(nrows=1, ncols=len(markers))
#         plt.suptitle('Train')
#         fig_test, axes_test = plt.subplots(nrows=1, ncols=len(markers))
#         plt.suptitle('Test')
#         for i, index_marker in enumerate(indices_markers):
#             marker_name = present_markers[index_marker]
#             coefficient = coefficients[index_marker]
#
#             X_train_augm_markers = augmented_data.X_train_augmented[:, index_marker]
#             X_test_augm_markers = augmented_data.X_test_augmented[:, index_marker]
#
#             xrange_train = np.linspace(min(X_train_augm_markers), max(X_train_augm_markers))
#             xrange_test = np.linspace(min(X_test_augm_markers), max(X_test_augm_markers))
#
#             axes_train[i].scatter(X_train_augm_markers, labels_train, color=colors_train)
#             axes_train[i].plot(xrange_train, 1 / (1 + np.exp(-(coefficient*xrange_train))), color='darkgray', lw=2)
#             axes_train[i].set_title(marker_name)
#             axes_train[i].set_yticks([0, 1])
#
#             axes_test[i].scatter(X_test_augm_markers, labels_test, color=colors_test)
#             axes_test[i].plot(xrange_test, 1 / (1 + np.exp(-(coefficient * xrange_test))), color='darkgray', lw=2)
#             axes_test[i].set_title(marker_name)
#             axes_test[i].set_yticks([0, 1])
#
#         plt.legend(handles=[h1, h2], loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.tight_layout()
#         plt.show()











# TODO: Make this function work (?)
# def plot_for_experimental_mixture_data(X_mixtures, y_mixtures, y_mixtures_matrix, inv_test_map, classes_to_evaluate,
#                                        mixtures_in_classes_of_interest, n_single_cell_types_no_penile, dists):
#     """
#     for each mixture category that we have measurements on, plot the
#     distribution of marginal LRs for each cell type, as well as for the special
#     combinations (eg vaginal+menstrual) also plot LRs as a function of distance
#     to nearest data point also plot experimental measurements together with LRs
#     found and distance in a large matrix plot
#
#     :param X_mixtures: N_experimental_mixture_samples x N_markers array of
#         observations
#     :param y_mixtures: N_experimental_mixture_samples array of int mixture labels
#     :param y_mixtures_matrix:  N_experimental_mixture_samples x
#         (N_single_cell_types + N_combos) n_hot encoding
#     :param inv_test_map: dict: mixture label -> mixture name
#     :param classes_to_evaluate: list of str, classes to evaluate
#     :param mixtures_in_classes_of_interest:  list of lists, specifying for each
#         class in classes_to_evaluate which mixture labels contain these
#     :param n_single_cell_types_no_penile: int: number of single cell types
#         excluding penile skin
#     :param dists: N_experimental_mixture_samples iterable of distances to
#         nearest augmented data point. Indication of whether the point may be an
#         outlier (eg measurement error or problem with augmentation scheme)
#     """
#     y_prob = model.predict_lrs(X_mixtures, )
#     y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
#         y_prob, mixtures_in_classes_of_interest, classes_map_updated, MAX_LR)
#
#     log_lrs_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
#     plt.subplots(3, 3, figsize=(18, 9))
#     for i, i_clas in enumerate(set(y_mixtures)):
#         indices_experiments = [j for j in range(len(y_mixtures)) if y_mixtures[j] == i_clas]
#         plt.subplot(3, 3, i + 1)
#         plt.xlim([-MAX_LR - .5, MAX_LR + .5])
#         bplot = plt.boxplot(log_lrs_per_class[indices_experiments, :], vert=False,
#                             labels=classes_to_evaluate, patch_artist=True)
#
#         for j, (patch, cla) in enumerate(zip(bplot['boxes'], classes_to_evaluate)):
#             if j < n_single_cell_types_no_penile:
#                 # single cell type
#                 if cla in inv_test_map[i_clas]:
#                     patch.set_facecolor('black')
#             else:
#                 # sample 'Vaginal.mucosa and/or Menstrual.secretion'
#                 for comb_class in cla.split(' and/or '):
#                     if comb_class in inv_test_map[i_clas]:
#                         patch.set_facecolor('black')
#         plt.title(inv_test_map[i_clas])
#     plt.savefig('mixtures_boxplot')
#
#     plt.subplots(3, 3, figsize=(18, 9))
#     for i in range(y_mixtures_matrix.shape[1]):
#         plt.subplot(3, 3, i + 1)
#         plt.ylim([-MAX_LR - .5, MAX_LR + .5])
#         plt.scatter(
#             dists + np.random.random(len(dists)) / 20,
#             log_lrs_per_class[:, i],
#             color=['red' if iv else 'blue' for iv in y_mixtures_matrix[:, i]],
#             alpha=0.1
#         )
#         plt.ylabel('LR')
#         plt.xlabel('distance to nearest data point')
#         plt.title(classes_to_evaluate[i])
#     plt.savefig('LRs_as_a_function_of_distance')
#
#     plt.figure()
#     plt.matshow(
#         np.append(
#             np.append(X_mixtures, log_lrs_per_class, axis=1),
#             np.expand_dims(np.array([d*5 for d in dists]), axis=1),
#             axis=1))
#     plt.savefig('mixtures binned data and log lrs')

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


# TODO: Make this funtion work?
# def plot_data(X):
#     """
#     plots the raw data points
#
#     :param X: N_samples x N_observations_per_sample x N_markers measurements
#     """
#     plt.matshow(combine_samples(X))
#     plt.savefig('single_cell_type_measurements_after_QC')
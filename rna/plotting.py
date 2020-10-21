"""
Plotting functions.
"""

import math
import os
import string

import matplotlib

import scipy
import numpy as np
import pandas as pd
from lir import Xn_to_Xy, plot, plot_score_distribution_and_calibrator_fit, Xy_to_Xn

from matplotlib import rc, pyplot as plt, patches as mpatches
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy.interpolate import interp1d
import seaborn as sns

from rna import constants
from rna.constants import celltype_specific_markers, DEBUG, COLWIDTH
from rna.utils import vec2string, prior2string, bool2str_binarize, bool2str_softmax

from lir.calibration import IsotonicCalibrator, LogitCalibrator

rc('text', usetex=False)


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
        xlabel = '10logLR (no calibration)'
        min_val = -4
        max_val = 4
    else:
        data = lr / (1 + lr)
        xlabel = 'Probability (no calibration)'
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
    uncalibrated_lrs = np.ravel(sorted(data))
    calibrated_lrs = calibrator.transform(uncalibrated_lrs.reshape(-1,1))
    axes[0, 1].hist(data1, color='orange', density=True, bins=30, label="h1", alpha=0.5)
    axes[0, 1].hist(data2, color='blue', density=True, bins=30, label="h2", alpha=0.5)
    axes[0, 1].plot(uncalibrated_lrs, calibrated_lrs / ( 1 + calibrated_lrs), color='orange', label='p1')
    axes[0, 1].plot(uncalibrated_lrs, 1/ ( 1 + calibrated_lrs), color='blue', label='p2')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlim(min_val, max_val)

    # plt.figure(figsize=(10, 10), dpi=100)
    # x = np.arange(-10, 10, .1)
    # calibrator.transform(x)
    # plt.hist(data1, bins=20, alpha=.25, density=True, label='class 0')
    # plt.hist(data2, bins=20, alpha=.25, density=True, label='class 1')
    # plt.plot(x, calibrator.p1, label='fit class 1')
    # plt.plot(x, calibrator.p0, label='fit class 0')
    # plt.savefig(f'calib_test_{celltype}')
    # plt.close()
    #
    # return

    # 3 KDE curves
    axes[1, 0].plot(uncalibrated_lrs,  calibrated_lrs / ( 1 + calibrated_lrs), color='orange', label='p1')
    axes[1, 0].plot(uncalibrated_lrs, 1/ ( 1 + calibrated_lrs), color='blue', label='p2')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_xlim(min_val, max_val)
    axes[1, 0].legend(loc='upper right')

    # 4 Ratio of two curves
    if calibration_on_loglrs:
        X_abovemin10 = np.unique(np.linspace(min_val, min(uncalibrated_lrs), 200))

        ratio_abovemin10 = calibrator.transform(X_abovemin10.reshape(-1,1))

        X_below10 = np.unique(np.linspace(max(uncalibrated_lrs), max_val, 200))

        ratio_below10 = calibrator.transform(X_below10.reshape(-1,1))

    axes[1, 1].set_xlim(min_val, max_val)
    axes[1, 1].plot(uncalibrated_lrs, calibrated_lrs, color='green', label='ratio')
    if calibration_on_loglrs:
        axes[1, 1].plot(X_abovemin10, ratio_abovemin10, color='green', linestyle=':', linewidth=1)
        axes[1, 1].plot(X_below10, ratio_below10, color='green', linestyle=':', linewidth=1)
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel('LR')

    # plot(uncalibrated_lrs, true_lrs, show_scatter=True, on_screen=False, path='scratch/pav.png')

    # 5
    axes[2, 0].plot(uncalibrated_lrs, np.log10(calibrated_lrs), color='green', label='ratio')
    if calibration_on_loglrs:
        axes[2, 0].plot(X_abovemin10, np.log10(ratio_abovemin10), color='green', linestyle=':', linewidth=1)
        axes[2, 0].plot(X_below10, np.log10(ratio_below10), color='green', linestyle=':', linewidth=1)
    axes[2, 0].set_xlabel(xlabel)
    axes[2, 0].set_ylabel('10log LR')
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
        axes[2, 1].set_xlim(max(-10,min(min_vals)), min(10,max(max_vals)))
        axes[2, 1].set_ylim(max(-10,min(min_vals)), min(10,max(max_vals)))
        axes[2, 1].set_xlabel("True 10logLRs before")
        axes[2, 1].set_ylabel("True 10logLRs after")
        axes[2, 1].legend(handles=[h1, h2], loc='upper right')

    # 7
    uncalibrated_lrs = calibrator.transform(data.reshape(-1,1))

    if calibration_on_loglrs:
        calibrated_data = np.log10(uncalibrated_lrs)
        xlabel = 'Calibrated 10logLR'
    else:
        calibrated_data = uncalibrated_lrs / (1 + uncalibrated_lrs)
        xlabel = 'Calibrated probability'

    calibrated_data[calibrated_data==np.inf] = 10
    calibrated_data[calibrated_data==-np.inf] = -10


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


def plot_property_all_lrs_all_folds(lrs_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                    kind='histogram',  # one of histogram, roc
                                    show=None, savefig=None):

    for method in lrs_for_all_methods.keys():

        plot_property_lrs_all_folds(lrs_for_all_methods[method], y_nhot_for_all_methods[method], target_classes,
                                    label_encoder, kind)

        if savefig is not None:
            matplotlib.rcParams['text.usetex'] = False
            plt.tight_layout()
            plt.savefig(savefig + '_' + method+'.png')
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close()


def plot_property_lrs_all_folds(lrs, y_nhot, target_classes, label_encoder,
                                kind, n_bins=30, title='after', density=True):

    loglrs = np.log10(lrs)
    n_target_classes = len(target_classes)
    if n_target_classes > 1:
        n_rows = math.ceil(n_target_classes / 2)
        if title == 'after':
            fig, axs = plt.subplots(n_rows, 2, figsize=(COLWIDTH*2, COLWIDTH*2), sharex=True, sharey=False)
        else:
            fig, axs = plt.subplots(n_rows, 2, figsize=(COLWIDTH*2, COLWIDTH*2*2/3), sharex=True, sharey=True)

    j = 0
    k = 0

    for t, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        loglrs1 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1), t]
        loglrs2 = loglrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0), t]
        # at some point, lir should support this (with an axis handle):
        # loglrs_, y = Xn_to_Xy(loglrs1, loglrs2)
        # plot_log_lr_distributions(loglrs_, y, savefig = f'lrs_hist_{target_class}.png')
        # plot_log_lr_distributions(loglrs_, y, kind='tippett', savefig = f'lrs_tippet_{target_class}.png')
        if n_target_classes == 1:
            if kind != 'histogram':
                return
                raise NotImplemented
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
            if kind != 'histogram':
                return
                raise NotImplemented
            axs[t].hist(loglrs1, color='orange', density=density, bins=n_bins, label="h1", alpha=0.5)
            axs[t].hist(loglrs2, color='blue', density=density, bins=n_bins, label="h2", alpha=0.5)
            axs[t].set_title(celltype)

            handles, labels = axs[0].get_legend_handles_labels()

            fig.text(0.5, 0.002, "10logLR", ha='center')
            if density:
                fig.text(0.002, 0.5, "Density", va='center', rotation='vertical')
            else:
                fig.text(0.002, 0.5, "Frequency", va='center', rotation='vertical')

            fig.legend(handles, labels, 'lower right')

        elif n_rows > 1:
            if kind == 'histogram':
                # plotting Tippett here
                # axs[j, k].hist(loglrs1, color='orange', histtype='step', cumulative=-1, density=density, bins=n_bins, label="h1")
                # axs[j, k].hist(loglrs2, color='blue', histtype='step', cumulative=-1, density=density, bins=n_bins, label="h2")
                axs[j, k].axvline(0, color='k', linestyle='--')

                n = np.arange(1, len(loglrs2) + 1) / np.float(len(loglrs2))
                Xs = np.sort(loglrs2.squeeze())[::-1]
                axs[j, k].step(Xs, n, color='blue', label='H2', alpha=.5)

                n = np.arange(1, len(loglrs1) + 1) / np.float(len(loglrs1))
                Xs = np.sort(loglrs1.squeeze())[::-1]
                axs[j, k].step(Xs, n, color='orange', label='H1', alpha=.5)

                x_label="10log(LR)"
                if density:
                    y_label="Cumulative density"
                else:
                    y_label="Frequency"

                handles, labels = axs[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, 'center right')

            elif kind=='roc':
                auc=plot_roc(*Xn_to_Xy(loglrs2, loglrs1), axs[j,k])
                x_label = 'False positive rate'
                y_label = 'True positive rate'
            else:
                raise ValueError(f'unknown plot kind {kind}')
            title = celltype.lower().replace('.', ' ').replace('menstrual secretion', 'MS').replace(
                'semen fertile and/or semen sterile', 'semen\n(sterile and/or fertile)').replace(
                'vaginal mucosa', 'VM')
            if kind=='roc' and auc:
                title += f'\nAUC={auc:.2f}, FP={np.mean(loglrs2 > 0):.2f}, FN={np.mean(loglrs1 < 0):.2f}'
            axs[j, k].set_title(title, fontdict={'fontsize': 12})

            if (t % 2) == 0:
                k = 1
            else:
                k = 0
                j = j + 1

            fig.text(0.5, 0.004, x_label, ha='center')
            fig.text(0.002, 0.5, y_label, va='center', rotation='vertical')




def plot_scatterplots_all_lrs_different_priors(lrs_for_all_methods, y_nhot_for_all_methods, target_classes,
                                               label_encoder, show=None, savefig=None):

    methods_no_prior = []
    for full_method_name in lrs_for_all_methods.keys():
        methods_no_prior.append(full_method_name.split("'")[1])
    methods_no_prior = np.unique(methods_no_prior).tolist()

    test_dict = OrderedDict()
    for full_method_name in methods_no_prior:
        for names in lrs_for_all_methods.keys():
            if full_method_name in names:
                if full_method_name in test_dict:
                    test_dict[full_method_name].update(
                        {names: (lrs_for_all_methods[names],
                                 y_nhot_for_all_methods[names])})
                else:
                    test_dict[full_method_name] = \
                        {names: (lrs_for_all_methods[names],
                                 y_nhot_for_all_methods[names])}

    loglrs1_list =[]
    loglrs2_list =[]
    methods_list =[]
    labels_list = []
    priors_list = []
    for t, target_class in enumerate(target_classes):
        for method, values in test_dict.items():
            loglrs = OrderedDict()
            y_nhot = OrderedDict()
            full_name = []
            priors = []
            for full_method_name, data in values.items():
                loglrs[full_method_name] = np.log10(data[0][:, t])
                y_nhot[full_method_name] = data[1]
                full_name.append(full_method_name)

                priors.append('[' + full_method_name.split('[')[2])
            assert np.array_equal(y_nhot[full_name[0]], y_nhot[full_name[1]])

            # fig, axes = plt.subplots(nrows=len(priors)//2, ncols=1, figsize=(16, 10))

            # target_class = np.reshape(target_class, (-1, 1))
            labels = np.max(np.multiply(y_nhot[full_name[0]], target_class), axis=1)

            for i_prior in range(len(priors)-1):
                # convoluted way of saving the uniform prior results many
                # times, to easily plot later
                if any(str([1] * len(target_class)) in x for x in priors):
                    index1 = priors.index(str([1] * len(target_class)))
                    index2 = i_prior
                    if index2>index1:
                        index2+=1
                    if index2 == index1:
                        continue
                    loglrs1 = loglrs[full_name[index1]]
                    loglrs2 = loglrs[full_name[index2]]
                else:
                    raise ValueError('should have baseline uniform prior to '
                                     'plot')

                loglrs1_list += list(loglrs1)
                loglrs2_list += list(loglrs2)

                methods_list += [method] * len(labels)
                labels_list += list(labels.squeeze())
                priors_list += [prior2string(priors[index2], label_encoder)] \
                               * len(labels)
        fig = plt.plot(figsize=(COLWIDTH * 2, COLWIDTH * 2 * 5 / 3))
        df = pd.DataFrame({'label': labels_list, 'method': methods_list,
                           'baseline': priors_list,
                           'log(LR) uniform': loglrs1_list,
                           'log(LR) adjusted baseline': loglrs2_list,
                           })
        sns.set(font_scale=1.5, rc={'text.usetex': False})

        grid = sns.relplot(data=df, x='log(LR) uniform', y='log(LR) adjusted baseline',
                       kind='scatter', col='baseline', row='method', hue='label',
                       style='label', legend=False,
                       facet_kws={'margin_titles': True})

        grid.map(sns.lineplot, y=[-15, 15], x=[-15, 15])
        [plt.setp(ax.texts, text="") for ax in
         grid.axes.flat]  # remove the original texts

        plt.setp(grid.fig.texts, text="")

        grid.set_titles(row_template='{row_name}', col_template='{col_name}')
        # grid.set_axis_labels('log(LR) - uniform background levels', 'log(LR) - adjusted background levels')
        plt.xlim(-5, 10)
        plt.xticks([-4,-2,0,2,4,6,8])
        plt.ylim(-5, 10)
        for i, axis in enumerate(grid.axes.flat):
            axis.text(-.0, 1.0, string.ascii_uppercase[i], transform=axis.transAxes,
                      size=20, weight='bold')

        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plt.text(0.5, 0.004, 'log(LR) - uniform background levels', ha='center')
        plt.text(0.002, 0.5, 'log(LR) - adjusted background levels', va='center', rotation='vertical')

        if savefig is not None:
            # plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save, dpi=constants.DPI)
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()


def plot_boxplot_of_metric(binarize, softmax, models, priors, n_metric, label_encoder, name_metric, prior_to_plot=None, savefig=None, show=None, ylim=[0,1]):
    def int2string_models(int, specify_which=None):
        if specify_which == None:
            raise ValueError("type must be set: 1 = transformation, 2 = probscalculations, 3 = model, 4 = prior")
        elif specify_which == 1:
            for i in range(len(binarize)):
                if int == i:
                    return bool2str_binarize(binarize[i])
        elif specify_which == 2:
            for i in range(len(softmax)):
                if int == i:
                    return bool2str_softmax(softmax[i])
        elif specify_which == 3:
            for i in range(len(models)):
                if int == i:
                    return models[i][0] if models[i][1] else models[i][0]+' uncal'
        elif specify_which == 4:
            for i in range(len(priors)):
                if int == i:
                    return prior2string(str(priors[i]), label_encoder)
        else:
            raise ValueError("Value {} for variable 'specify which' does not exist".format(specify_which))


    if not prior_to_plot:
        prior_to_plot=priors[0]

    n_per_fold = n_metric.shape[0]
    i_transformations = n_metric.shape[1]
    j_probscalulations = n_metric.shape[2]
    k_models = n_metric.shape[3]
    p_priors = n_metric.shape[4]

    fig = plt.figure(figsize=(COLWIDTH*2, COLWIDTH))
    ax = plt.subplot(111)
    trans_list = []
    probs_list = []
    models_list = []
    priors_list = []
    metric_list = []
    for i in range(i_transformations):
        for j in range(j_probscalulations):
            for k in range(k_models):
                for p in range(p_priors):
                    if priors[p] == prior_to_plot:
                        trans_list+=[int2string_models(i, 1)]*n_per_fold
                        probs_list+=[int2string_models(j, 2)]*n_per_fold
                        models_list+=[int2string_models(k, 3)]*n_per_fold
                        priors_list+=[int2string_models(p, 4)]*n_per_fold
                        metric_list+=list(n_metric[:, i, j, k, p].squeeze())


    df = pd.DataFrame({
        'multi-label strategy': probs_list,
        'values': trans_list,
        'prior': priors_list,
        'model': models_list,
        name_metric: metric_list})
    sns.set(font_scale=1.5, rc={'text.usetex': False})
    g = sns.factorplot(data=df, x='multi-label strategy', y=name_metric,
               hue='model', col='values',
               kind='box', legend=True, legend_out =True, ci=None)
    plt.ylim(ylim)

    for i,axis in enumerate(g.axes.flat):
        axis.text(-.01,1.05, string.ascii_uppercase[i], transform=axis.transAxes,
            size=25, weight='bold')

    if savefig is not None:
        # plt.tight_layout()
        plt.savefig(savefig, dpi=constants.DPI)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()

    plt.close(fig)



def plot_progress_of_metric(binarize, softmax, models, priors, n_metric, label_encoder, name_metric, savefig=None, show=None):

    def int2string_models(int, specify_which=None):
        if specify_which == None:
            raise ValueError("type must be set: 1 = transformation, 2 = probscalculations, 3 = model, 4 = prior")
        elif specify_which == 1:
            for i in range(len(binarize)):
                if int == i:
                    return bool2str_binarize(binarize[i])
        elif specify_which == 2:
            for i in range(len(softmax)):
                if int == i:
                    return bool2str_softmax(softmax[i])
        elif specify_which == 3:
            for i in range(len(models)):
                if int == i:
                    return models[i][0]
        elif specify_which == 4:
            for i in range(len(priors)):
                if int == i:
                    return prior2string(str(priors[i]), label_encoder)
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
        if i is not 1:
            for j in range(j_probscalulations):
                for k in range(k_models):
                    if k is not 2:
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
        plt.savefig(savefig, dpi=constants.DPI)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def plot_pavs_all_methods(lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes,
                          label_encoder, show=None, savefig=None):

    for method in lrs_before_for_all_methods.keys():
        plot_pavs(lrs_before_for_all_methods[method], lrs_after_for_all_methods[method], y_nhot_for_all_methods[method],
                  target_classes, label_encoder, method_name=method, show=show, savefig=savefig)


def plot_pavs(lrs_before, lrs_after, y_nhot, target_classes, label_encoder, method_name, savefig=None, show=None):

    for t, target_class in enumerate(target_classes):

        loglrs_before = np.log10(lrs_before[:, t])
        loglrs_after = np.log10(lrs_after[:, t])
        labels = np.max(np.multiply(y_nhot, target_class), axis=1)

        if not any(labels) or all(labels):
            # cant do much with only positive or negative labels
            continue


        if np.array_equal(loglrs_before, loglrs_after):
            fig, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
            plot_pav(loglrs_before, labels, 'Before calibration', axs1)
        else:
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

        if method_name is not None:
            target_class_save =  method_name + '_' + target_class_save

        plot(lrs_after[:, t], labels, show_scatter=True, on_screen=False,
             path=savefig + '_' + target_class_save+'_after')

        plot(lrs_before[:, t], labels, show_scatter=True, on_screen=False,
             path=savefig + '_' + target_class_save+'_before')


def plot_pav(loglr, labels, title, ax, show_scatter=True):
    """
    Plots pav plots for all cell types before and after calibration.
    :param title:
    """

    ax=ax

    try:
        pav = IsotonicCalibrator()
        pav_loglrs = pav.fit_transform(loglr, labels)
        xrange = [-10, 10]

        ax.axis('equal')
        ax.axis(xrange + xrange)
        ax.plot(xrange, xrange, color='black')

        pav_x = np.arange(*xrange, .01)
        pav_transformed_x = pav.transform(pav_x)
        pav_transformed_x = np.where(pav_transformed_x == -np.Inf, -11, pav_transformed_x)
        pav_transformed_x = np.where(pav_transformed_x == np.Inf, 11, pav_transformed_x)
        ax.plot(pav_x, pav_transformed_x, color='green')
        ax.grid(True, linestyle=':')
        ax.set_title(title)
        if show_scatter:
            indices_h1 = np.argwhere(labels == 1).ravel()
            indices_h0 = np.argwhere(labels == 0).ravel()
            h1 = mpatches.Patch(color='orange', label='h1')
            h2 = mpatches.Patch(color='blue', label='h2')
            ax.scatter(loglr[indices_h1], [-9]*len(pav_loglrs[indices_h1]), color='orange', s=1.5, marker='+')
            ax.scatter(loglr[indices_h0], [-9.5]*len(pav_loglrs[indices_h0]), color='blue', s=1.5, marker='x')
            ax.legend(handles=[h1, h2], loc='upper right')
        ax.set_xticks(np.linspace(-10, 10, 9))
        ax.set_yticks(np.linspace(-10, 10, 9))

        return ax

    except ValueError:
        # when the target class does not exist in the test data
        pass


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
        ax.plot(np.log10(sim_lrs), sim_punish_d, color='blue')

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
        plt.savefig(savefig, dpi=constants.DPI)
        plt.close()
    if show or savefig is None:
        plt.tight_layout()
        plt.show()


def derivative_of_roc_is_lr(lrs_all_methods, y_nhot_all_methods, fpr, tpr, t, target_class):
    from statsmodels.nonparametric.kernel_regression import KernelReg
    from scipy.misc import derivative

    loglrs = np.log10(lrs_all_methods['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'][:, t])
    y_nhot = y_nhot_all_methods['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]']
    labels = np.max(np.multiply(y_nhot, target_class), axis=1)
    loglrs_h1 = loglrs[np.argwhere(labels == 1)]
    loglrs_h2 = loglrs[np.argwhere(labels == 0)]

    # Make a function from coordinates (fpr, tpr) to enable calculating the differtiated function
    '''
    Interpolation is the process of finding a value between two points on a line or a curve. To help us remember 
    what it means, we should think of the first part of the word, 'inter,' as meaning 'enter,' which reminds us to 
    look 'inside' the data we originally had. This tool, interpolation, is not only useful in statistics, but is also 
    useful in science, business, or when there is a need to predict values that fall within two existing data points.
    '''
    f = interp1d(fpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'], tpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'], kind='linear')
    # When score is 1
    x = np.linspace(fpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'].min(), fpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'].max(),
                    len(fpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]']))
    y = f(x)
    plt.plot(x, y)
    plt.show()

    dy = np.zeros(x.shape[0])
    dy[0:-1] = np.diff(y) / np.diff(x)
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    # # dy = np.where(dy == np.Inf, 10, dy)
    # # dy = np.where(dy == -np.Inf, -10, dy)
    plt.scatter(x, dy)
    plt.show()

    kr = KernelReg(y, x, 'c')
    y_pred, y_std = kr.fit(x)

    # create cumulative density function
    F_d = scipy.stats.norm.cdf(loglrs_h2)
    F_d_ = 1 - F_d
    f_F_d_ = interp1d(loglrs_h2.ravel(), F_d_.ravel())
    fig, ax = plt.subplots()
    fitx = np.linspace(loglrs_h2.min(), loglrs_h2.max(), 100)
    ax.scatter(loglrs_h2, F_d_)
    ax.plot(fitx, f_F_d_(fitx), color='black')
    fig.show()


    logscore = 0
    T = f_F_d_(logscore).max()
    true_LR = derivative(f, T, dx=1e-2)
    fig, ax = plt.subplots()
    ax.scatter(fpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'], tpr['bin_sig_MLR_[1, 1, 1, 1, 1, 1, 1, 1]'],
               color='orange', alpha=0.2, label='true coordinates')
    ax.plot(x, f(x), color='black', lw=1, label='approximated function')
    # ax.plot(x, y_pred, color='green', lw=1, linstyle='--', label='smoothed function')
    ax.axvline(x=T, color='red', label='x (10logScore) = 0')
    ax.legend()
    ax.set_title('True LR at Fd({})={} is {}'.format(logscore, T, true_LR))
    fig.show()


def plot_roc(lrs, y_true, ax):

    fpr, tpr, _ = roc_curve(y_true, lrs)
    roc_auc = auc(fpr, tpr)

    lw=1.5
    ax.plot(fpr, tpr, color='k', lw=lw, linestyle='-',
                 label='AUC={:0.2f}'.format(roc_auc))
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    return roc_auc


def plot_coefficient_importances(model, target_classes, present_markers, label_encoder, savefig=None, show=None):
    plt.figure(figsize=(COLWIDTH, COLWIDTH))
    plt.figure(figsize=(5, 5))
    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        celltype = target_class_str.split(' and/or ')
        if DEBUG or len(celltype)>1:
            intercept, coefficients = model.get_coefficients(t, target_class)
            if not intercept:
                return
            plot_coefficient_importance(intercept, coefficients, present_markers, celltype)

            target_class_save = target_class_str.replace(" ", "_")
            target_class_save = target_class_save.replace(".", "_")
            target_class_save = target_class_save.replace("/", "_")

            if savefig is not None:
                plt.tight_layout()
                plt.savefig(savefig + '_' + target_class_save, dpi=constants.DPI)
            if show or savefig is None:
                plt.tight_layout()
                plt.show()

            plt.close()


def plot_coefficient_importance(intercept, coefficients, present_markers, celltypes):
    """

    :param intercept:
    :param coefficients: these are assumed to already be transformed to base 10
    :param present_markers:
    :param celltypes: list of strings: list of celltypes
    :return:
    """

    def calculate_max_base_log_lr(intercept, coefficients):
        return np.sum([coef for coef in coefficients.squeeze() if coef > 0] + [intercept])

    coefficients = np.reshape(coefficients, -1)

    max_base = calculate_max_base_log_lr(intercept, coefficients)

    # sort
    sorted_indices = np.argsort(coefficients)
    coefficients = coefficients[sorted_indices]
    present_markers = np.array(present_markers)[sorted_indices].tolist()
    x = np.linspace(1, len(coefficients)+2, len(coefficients)+2)

    # get the indices of the celltype specific markers
    marker_indices = []
    for celltype in celltypes:
        for marker in celltype_specific_markers[celltype]:
            if marker is not None:
                marker_indices.append(present_markers.index(marker))
    marker_indices = np.unique(marker_indices)

    barlist = plt.barh(list(x), [intercept, 0] + list(coefficients), color='grey', alpha=0.6, label='other')
    for marker_index in marker_indices:
        # highlight the markers that are celltype specific
        barlist[marker_index+2].set_color('navy')
        barlist[marker_index+2].set_hatch("/")
    try:
        barlist[marker_indices[0]+2].set_label('body fluid specific')
    except IndexError:
        pass
    plt.yticks(x, ['intercept', ''] + present_markers)
    if DEBUG:
        plt.title('Max, base 10log LR = {:.1f}, {:.1f}'.format(max_base, intercept))
    plt.xlabel('Coefficient')
    if not DEBUG:
        # to get same axes on penile/no penile
        plt.xlim([-2, 3.3])
    plt.ylabel('Marker names')

    # plt.legend(loc='lower right')


def plot_lrs_with_bootstrap_ci(lrs_after_calib, all_lrs_after_calib_bs, target_classes, label_encoder, show=None,
                               savefig=None):

    def confidence_interval(all_lrs_after_calib_bs_tc, alpha):

        lower_bounds = np.percentile(all_lrs_after_calib_bs_tc, (alpha/2)*100, axis=1)
        upper_bounds = np.percentile(all_lrs_after_calib_bs_tc, (1 - (alpha/2))*100, axis=1)

        return lower_bounds, upper_bounds

    lower_bounds_tc = dict()
    upper_bounds_tc = dict()
    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        lower_bounds, upper_bounds = confidence_interval(all_lrs_after_calib_bs[:, t, :], alpha=0.05)
        lower_bounds_tc[target_class_str] = lower_bounds
        upper_bounds_tc[target_class_str] = upper_bounds

        plot_lrs_with_bootstrap_ci_per_target_class(lrs_after_calib[:, t], lower_bounds, upper_bounds)

        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + target_class_save, dpi=constants.DPI)
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close()

    return lower_bounds_tc, upper_bounds_tc


def plot_lrs_with_bootstrap_ci_per_target_class(lrs, lower_bounds, upper_bounds):

    X = np.arange(lrs.shape[0])
    sorted_indices = np.argsort(lrs)

    llrs = np.log10(lrs)[sorted_indices]
    llower_bounds = np.log10(lower_bounds)[sorted_indices]
    lupper_bounds = np.log10(upper_bounds)[sorted_indices]

    plt.plot(X, llrs, color='black')
    plt.fill_between(X, llower_bounds, lupper_bounds, color='gray')
    plt.xlabel('number of samples in test data')
    plt.ylabel('10logLR')


def plot_sankey_data():
    """
    no actual analysis, just visualising precomputed numbers
    :return:
    """
    import plotly.graph_objects as go

    node_colors = [[0,0,0],
                   [255, 100, 60], [50, 50, 255], [100, 100, 100],
                   [100,100,100],
                   [200, 120, 100], [220, 120, 90], [255,120,80], [50, 50, 255]]

    sources = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
    targets = [1, 2, 3, 6, 7, 4, 8, 4, 5, 8, 6]
    values = [34, 23, 29, 2, 32, 1, 22, 2, 8, 16, 3]
    fig = go.Figure(data=[go.Sankey(arrangement='perpendicular',
        node=dict(
            # pad = 15,
            # thickness = 20,
            # line = dict(color = "black", width = 2),
            label=[f"{sum(values[:3])} traces",
                   "indication for presence of ...", "no indication for presence of ...",
                   "no reliable statement possible",
                   'LR 0.5-2',
                   'LR 2-10: weak support', "LR 10-100: moderate support", 'LR>100: moderately strong support', 'LR<0.5'],
            color=[f'rgba({col[0]}, '
                   f'{col[1]}, '
                   f'{col[2]}, 1)'
                   for col in node_colors]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=[f'rgba({node_colors[target][0]}, '
                   f'{node_colors[target][1]}, '
                   f'{node_colors[target][2]}, .2)'
                   for target in targets],
            label=values
        ))])

    # fig.update_layout(
    #     font=dict(size = 10, color = 'white'),
    #     plot_bgcolor='black',
    #     paper_bgcolor='black'
    # )
    fig.show(width=100)


def calibration_example(savepath):
    factor = 10
    uncalibrated_lrs_h1 = np.array(
        [.01 * factor] + [.1 * factor]*10 + [1 * factor]*100 + [10 * factor] * 100 + [100 * factor] * 100)
    uncalibrated_lrs_h2 = factor / (uncalibrated_lrs_h1/factor)
    X, y = Xn_to_Xy(uncalibrated_lrs_h2, uncalibrated_lrs_h1)
    X = np.log10(X)
    calibrator = LogitCalibrator()
    calibrator.fit(X, y)
    fig, ax = plt.subplots(2, 2, figsize=(COLWIDTH*2*2, COLWIDTH*2))
    x = np.arange(-1, 3, .01)
    transformed = calibrator.transform(x)
    points0, points1 = Xy_to_Xn(X, y)
    points = np.unique(points0)
    ax[0,0].hist(points1, bins=sorted([p-.01 for p in points]+[p+.1 for p in points]),
                 alpha=.7, density=False, label='H_1', color='orange')
    ax[0,0].hist(points0, bins=sorted([p-.1 for p in points]+[p+.01 for p in points]),
                 alpha=.7, density=False, label='H_2', color='blue')
    ax[0,0].set_xlabel('log(s)')
    ax[0,0].set_ylabel('density')
    ax[0,0].set_yticks([])

    ax[0,1].scatter(points1+np.random.random(points1.size)/5-0.1, np.ones(points1.shape), s=5, color='orange')
    ax[0,1].scatter(points0+np.random.random(points0.size)/5-0.1, np.zeros(points0.shape), s=5, color='blue')
    ax[0,1].plot(x, calibrator.p1, color='k')
    ax[0,1].set_yticks([0,1])
    ax[0,1].set_xlabel('log(s)')
    ax[0,1].set_ylabel('logistic regression fit')

    points0, points1 = Xy_to_Xn(np.log10(calibrator.transform(X)), y)
    points = np.unique(points0)
    ax[1,1].hist(points1, bins=sorted([p-.01 for p in points]+[p+.1 for p in points]),
                 alpha=.7, density=False, label='H_1', color='orange')
    ax[1,1].hist(points0, bins=sorted([p-.1 for p in points]+[p+.01 for p in points]),
                 alpha=.7, density=False, label='H_2', color='blue')
    ax[1,1].set_xlabel('calibrated log(LR)')
    ax[1,1].set_ylabel('density')
    ax[1,1].set_xticks(points)
    ax[1,1].set_xticklabels([f'{p:2.2f}' for p in points])
    ax[1,1].set_yticks([])

    ax[1,0].plot(x, np.log10(transformed), color='k')
    ax[1,0].set_xlabel('log(s)')
    # ax[1,0].set_xlabel('uncalibrated log(LR)')
    ax[1,0].set_ylabel('calibrated log(LR)')
    ax[1,0].set_xticks([-1,0,1,2,3])

    for i,axis in enumerate(ax.flat):
        axis.text(-.1,1.1, string.ascii_uppercase[i], transform=axis.transAxes,
            size=20, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'calibration_example'), dpi=constants.DPI)
    plt.close()


def plot_multiclass_comparison(log_lrs, multi_log_lrs, sample, save_path):
    plt.style.use('ggplot')

    plt.figure(figsize=(COLWIDTH, COLWIDTH))
    df = pd.DataFrame({'multiclass log(LR)': multi_log_lrs,
                       'multi-label log(LR)': log_lrs[0]})
    p = sns.scatterplot(data=df, x='multiclass log(LR)',
                        y='multi-label log(LR)')
    # add annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p.text(multi_log_lrs[line], log_lrs[0][line],
               constants.single_cell_types_short[line],
               horizontalalignment='left',
               size='small', color='black')
    # make square plot
    ys = plt.ylim()
    xs = plt.xlim()
    maxs = [min(ys[0], xs[0]), max(ys[1], xs[1])]
    plt.ylim(maxs)
    plt.xlim(maxs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loglrs_for_' + str(sample)), dpi=constants.DPI)
    plt.close()
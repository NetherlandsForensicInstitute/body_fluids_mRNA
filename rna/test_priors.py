"""

"""

import os

import numpy as np
import rna.settings as settings
import matplotlib.pyplot as plt

from collections import OrderedDict

from sklearn.model_selection import train_test_split

from rna.analytics import combine_samples
from rna.augment import MultiLabelEncoder, augment_splitted_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import perform_analysis
from rna.utils import vec2string, string2vec
from rna.plotting import plot_scatterplot_lrs


def test_priors(nfolds, tc):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))


    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)


    # ======= Initialize =======
    emtpy_numpy_array = np.zeros((nfolds, len(settings.binarize), len(settings.softmax), len(settings.models), len(settings.priors)))
    accuracies_train, accuracies_test, accuracies_test_as_mixtures, accuracies_mixtures, accuracies_single,\
    cllr_test, cllr_test_as_mixtures, cllr_mixtures = [dict() for i in range(8)]

    for target_class in target_classes:
        target_class_str = vec2string(target_class, label_encoder)

        accuracies_train[target_class_str] = emtpy_numpy_array.copy()
        accuracies_test[target_class_str] = emtpy_numpy_array.copy()
        accuracies_test_as_mixtures[target_class_str] = emtpy_numpy_array.copy()
        accuracies_mixtures[target_class_str] = emtpy_numpy_array.copy()
        accuracies_single[target_class_str] = emtpy_numpy_array.copy()

        cllr_test[target_class_str] = emtpy_numpy_array.copy()
        cllr_test_as_mixtures[target_class_str] = emtpy_numpy_array.copy()
        cllr_mixtures[target_class_str] = emtpy_numpy_array.copy()

    lrs_for_model_per_fold = OrderedDict()
    for n in range(nfolds):
        print("Fold {}".format(n))

        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        lrs_for_model = OrderedDict()
        for i, binarize in enumerate(settings.binarize):
            print(" Binarize the data: {} {}".format(binarize, i))
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(settings.priors):
                print("Priors for augmenting data: {}".format(priors))

                augmented_data[str(priors)] = augment_splitted_data(X_calib, X_test, X_train, binarize, from_penile,
                                                                    label_encoder, n_celltypes, n_features, priors,
                                                                    y_calib, y_nhot_mixtures, y_test, y_train, AugmentedData)

            # ======= Transform data accordingly =======
            if binarize:
                X_train_transformed = np.where(combine_samples(X_train) > 150, 1, 0)
                X_calib_transformed = np.where(combine_samples(X_calib) > 150, 1, 0)
                X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            else:
                X_train_transformed = combine_samples(X_train) / 1000
                X_calib_transformed = combine_samples(X_calib) / 1000
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                print(" Use softmax to calculate probabilities with: {} {}".format(softmax, j))
                for k, models in enumerate(settings.models):
                    print(" Model: {} {}".format(models[0], k))

                    # ======= Calculate LRs before and after calibration =======
                    if settings.augment:
                        model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                        lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, y_test_as_mixtures_nhot_augmented, \
                        lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(augmented_data,
                                                                                                         X_mixtures,
                                                                                                         softmax,
                                                                                                         models, mle,
                                                                                                         target_classes)

                    key_name = str(binarize) + '_' + str(softmax) + '_' + str(models[0])
                    lrs_for_model[key_name] = LrsAfterCalib(lrs_after_calib, y_test_nhot_augmented,
                                                            lrs_test_as_mixtures_after_calib, y_test_as_mixtures_nhot_augmented,
                                                            lrs_after_calib_mixt)

                    # TODO: Check whether want to include
                    # else:
                    #     y_train_transformed = mle.inv_transform_single(y_train)
                    #     y_train_transformed = mle.labels_to_nhot(y_train_transformed)
                    #     y_calib_transformed = mle.inv_transform_single(y_calib)
                    #     y_calib_transformed = mle.labels_to_nhot(y_calib_transformed)
                    #     model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                    #     lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                    #         perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_transformed,
                    #                          y_train_transformed, X_calib_transformed, y_calib_transformed, X_test_augmented,
                    #                          y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes, save_hist=True)


                    # ======= Calculate performance metrics =======
                    # for t, target_class in enumerate(target_classes):
                    #     for p, priors in enumerate(settings.priors):
                    #         str_prior = str(priors)
                    #         target_class_str = vec2string(target_class, label_encoder)
                    #
                    #         accuracies_train[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                    #             model[str_prior], mle, augmented_data[str_prior].y_train_nhot_augmented,
                    #             augmented_data[str_prior].X_train_augmented, target_classes)[t]
                    #         accuracies_test[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                    #             model[str_prior], mle, augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented,
                    #             augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_augmented, target_classes)[t]
                    #         accuracies_test_as_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                    #             model[str_prior], mle, augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented,
                    #             augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_as_mixtures_augmented, target_classes)[t]
                    #         accuracies_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                    #             model[str_prior], mle, y_nhot_mixtures, X_mixtures, target_classes)[t]
                    #         accuracies_single[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                    #             model[str_prior], mle, mle.inv_transform_single(y_test), X_test_transformed, target_classes)[t]
                    #
                    #         cllr_test[target_class_str][n, i, j, k, p] = cllr(
                    #             lrs_after_calib[str_prior][:, t], augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented, target_class)
                    #         cllr_test_as_mixtures[target_class_str][n, i, j, k, p] = cllr(
                    #             lrs_test_as_mixtures_after_calib[str_prior][:, t], augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented, target_class)
                    #         cllr_mixtures[target_class_str][n, i, j, k, p] = cllr(
                    #             lrs_after_calib_mixt[str_prior][:, t], y_nhot_mixtures, target_class)

        lrs_for_model_per_fold[str(n)] = lrs_for_model

    lrs_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold)
    plot_scatterplot_lrs(lrs_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                         savefig=os.path.join('scratch', 'LRs_for_different_priors'))
    # plots_histogram_all_lrs(lrs_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder)


    plot_boxplot_of_metric(accuracies_train['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_train'))
    plot_boxplot_of_metric(accuracies_test['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_test'))
    plot_boxplot_of_metric(accuracies_test_as_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_test_as_mixtures'))
    plot_boxplot_of_metric(accuracies_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_mixtures'))
    plot_boxplot_of_metric(accuracies_single['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_single'))

    plot_boxplot_of_metric(cllr_test['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_test'))
    plot_boxplot_of_metric(cllr_test_as_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_test_as_mixtures'))
    plot_boxplot_of_metric(cllr_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_mixtures'))

# TODO: Want to change to dict?
class AugmentedData():

    def __init__(self, X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented, \
           X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented):
        self.X_train_augmented = X_train_augmented
        self.y_train_nhot_augmented = y_train_nhot_augmented
        self.X_calib_augmented = X_calib_augmented
        self.y_calib_nhot_augmented = y_calib_nhot_augmented
        self.X_test_augmented = X_test_augmented
        self.y_test_nhot_augmented = y_test_nhot_augmented
        self.X_test_as_mixtures_augmented = X_test_as_mixtures_augmented
        self.y_test_as_mixtures_nhot_augmented = y_test_as_mixtures_nhot_augmented

# TODO: Want to change to dict?
class LrsAfterCalib():

    def __init__(self, lrs_after_calib, y_test_nhot_augmented, lrs_test_as_mixtures_after_calib,
                 y_test_as_mixtures_nhot_augmented, lrs_after_calib_mixt):
        self.lrs_after_calib = lrs_after_calib
        self.y_test_nhot_augmented = y_test_nhot_augmented
        self.lrs_test_as_mixtures_after_calib = lrs_test_as_mixtures_after_calib
        self.y_test_as_mixtures_nhot_augmented = y_test_as_mixtures_nhot_augmented
        self.lrs_after_calib_mixt = lrs_after_calib_mixt


def calculate_lrs_for_different_priors(augmented_data, X_mixtures, softmax, models, mle, target_classes):

    # must be tested on the same test data
    X_test_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_augmented
    y_test_nhot_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented
    X_test_as_mixtures_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_as_mixtures_augmented
    y_test_as_mixtures_nhot_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented

    model = OrderedDict()
    lrs_before_calib = OrderedDict()
    lrs_after_calib = OrderedDict()
    lrs_test_as_mixtures_before_calib = OrderedDict()
    lrs_test_as_mixtures_after_calib = OrderedDict()
    lrs_before_calib_mixt = OrderedDict()
    lrs_after_calib_mixt = OrderedDict()

    for i, (key, data) in enumerate(augmented_data.items()):
        print(" Prior: {}".format(key))

        X_train_augmented = data.X_train_augmented
        y_train_nhot_augmented = data.y_train_nhot_augmented
        X_calib_augmented = data.X_calib_augmented
        y_calib_nhot_augmented = data.y_calib_nhot_augmented

        model_i, lrs_before_calib_i, lrs_after_calib_i, \
        lrs_test_as_mixtures_before_calib_i, lrs_test_as_mixtures_after_calib_i, \
        lrs_before_calib_mixt_i, lrs_after_calib_mixt_i = \
            perform_analysis(softmax, models, mle, X_train_augmented, y_train_nhot_augmented, X_calib_augmented,
                             y_calib_nhot_augmented, X_test_augmented, y_test_nhot_augmented,
                             X_test_as_mixtures_augmented, X_mixtures, target_classes)

        model[key] = model_i
        lrs_before_calib[key] = lrs_before_calib_i
        lrs_after_calib[key] = lrs_after_calib_i
        lrs_test_as_mixtures_before_calib[key] = lrs_test_as_mixtures_before_calib_i
        lrs_test_as_mixtures_after_calib[key] = lrs_test_as_mixtures_after_calib_i
        lrs_before_calib_mixt[key] = lrs_before_calib_mixt_i
        lrs_after_calib_mixt[key] = lrs_after_calib_mixt_i

    return model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
           lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, y_test_as_mixtures_nhot_augmented, \
           lrs_before_calib_mixt, lrs_after_calib_mixt


def combine_lrs_for_all_folds(lrs_for_model):
    """
    Combines the lrs calculated on test data for each fold.

    :param lrs_for_model:
    :return:
    """

    lrs_for_all_methods = dict()
    y_nhot_for_all_methods = dict()
    for fold, methods in lrs_for_model.items():

        for method, data in methods.items():
            priors = list(data.lrs_after_calib.keys())

            for prior in priors:
                prior_method = method + '_' + prior

                if prior_method in lrs_for_all_methods:
                    lrs_for_all_methods[prior_method] = np.append(lrs_for_all_methods[prior_method], data.lrs_after_calib[prior], axis=0)
                    y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method], data.y_test_nhot_augmented, axis=0)
                else:
                    lrs_for_all_methods[prior_method] = data.lrs_after_calib[prior]
                    y_nhot_for_all_methods[prior_method] = data.y_test_nhot_augmented

    return lrs_for_all_methods, y_nhot_for_all_methods


def plots_histogram_all_lrs(lrs_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder, show=None,
                            savefig=None):

    for method in lrs_for_all_methods.keys():

        plot_histogram_log_lr(lrs_for_all_methods[method], y_nhot_for_all_methods[method], target_classes, label_encoder)

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + method)
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()


def plot_histogram_log_lr(lrs, y_nhot, target_classes, label_encoder, n_bins=30, title='after', density=True):

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
"""

"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import rna.settings as settings

from collections import OrderedDict
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from rna.analytics import cllr, combine_samples, calculate_accuracy
from rna.augment import augment_data, MultiLabelEncoder, only_use_same_combinations_as_in_mixtures
from rna.lr_system import perform_analysis
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import vec2string, string2vec


def test_priors(nfolds, tc):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    for n in range(nfolds):
        start = time.time()

        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        lrs_for_model = OrderedDict()
        for i, binarize in enumerate(tqdm(settings.binarize)):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)

            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(tqdm(settings.priors)):

                augmented_data[str(priors)] = augment_splitted_data(X_calib, X_test, X_train, binarize, from_penile,
                                                                    label_encoder, n_celltypes, n_features, priors,
                                                                    y_calib, y_nhot_mixtures, y_test, y_train, AugmentedData)

            # ======= Convert data accordingly =======
            # if binarize:
            #     X_train_transformed = np.where(combine_samples(X_train) > 150, 1, 0)
            #     X_calib_transformed = np.where(combine_samples(X_calib) > 150, 1, 0)
            #     X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            # else:
            #     X_train_transformed = combine_samples(X_train) / 1000
            #     X_calib_transformed = combine_samples(X_calib) / 1000
            #     X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(tqdm(settings.softmax)):
                try:
                    for k, models in enumerate(tqdm(settings.models)):
                        time.sleep(3)
                        # print("Fold {} \n Binarize the data: {} \n Use softmax to calculate probabilities with: {} \n Model: {}".format(n, binarize, softmax, models[0]))

                        # ======= Calculate LRs before and after calibration =======
                        if settings.augment:
                            lrs_before_calib, lrs_after_calib, \
                            lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, \
                            lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(
                                augmented_data, X_mixtures, n, binarize, softmax, models, mle, label_encoder, target_classes)

                        key_name = str(binarize) + '_' + str(softmax) + '_' + str(models[0])
                        lrs_for_model[key_name] = LrsAfterCalib(lrs_after_calib, lrs_test_as_mixtures_after_calib, lrs_after_calib_mixt)

                        ### UNTIL HERE
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
                except:
                    # this is when the probabilities are calculated with the softmax function and the original data
                    # is used to train the models with.
                    continue

        end = time.time()
        print("Execution time in seconds: {}".format(end - start))

        plot_scatterplot_lrs(lrs_for_model)


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


class LrsAfterCalib():

    def __init__(self, lrs_after_calib, lrs_test_as_mixtures_after_calib, lrs_after_calib_mixt):
        self.lrs_after_calib = lrs_after_calib
        self.lrs_test_as_mixtures_after_calib = lrs_test_as_mixtures_after_calib
        self.lrs_after_calib_mixt = lrs_after_calib_mixt


def augment_splitted_data(X_calib, X_test, X_train, binarize, from_penile, label_encoder, n_celltypes, n_features,
                          priors, y_calib, y_nhot_mixtures, y_test, y_train, class_to_save):

    X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features,
                                                             settings.nsamples[0], label_encoder, priors,
                                                             binarize=binarize, from_penile=from_penile)
    X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features,
                                                             settings.nsamples[1], label_encoder, priors,
                                                             binarize=binarize, from_penile=from_penile)
    X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features,
                                                           settings.nsamples[2], label_encoder, priors,
                                                           binarize=binarize, from_penile=from_penile)
    X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented = only_use_same_combinations_as_in_mixtures(
        X_test_augmented, y_test_nhot_augmented, y_nhot_mixtures)

    class_to_return = class_to_save(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented, \
           X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented)

    return class_to_return


def calculate_lrs_for_different_priors(augmented_data, X_mixtures, n, binarize, softmax, models, mle, label_encoder,
                                       target_classes):

    X_test_augmented = augmented_data['None'].X_test_augmented
    y_test_nhot_augmented = augmented_data['None'].y_test_nhot_augmented
    X_test_as_mixtures_augmented = augmented_data['None'].X_test_as_mixtures_augmented

    lrs_before_calib = OrderedDict()
    lrs_after_calib = OrderedDict()
    lrs_test_as_mixtures_before_calib = OrderedDict()
    lrs_test_as_mixtures_after_calib = OrderedDict()
    lrs_before_calib_mixt = OrderedDict()
    lrs_after_calib_mixt = OrderedDict()

    for i, (key, data) in enumerate(augmented_data.items()):
        if key != 'None':

            X_train_augmented = data.X_train_augmented
            y_train_nhot_augmented = data.y_train_nhot_augmented
            X_calib_augmented = data.X_calib_augmented
            y_calib_nhot_augmented = data.y_calib_nhot_augmented

            model_i, lrs_before_calib_i, lrs_after_calib_i, \
            lrs_test_as_mixtures_before_calib_i, lrs_test_as_mixtures_after_calib_i, \
            lrs_before_calib_mixt_i, lrs_after_calib_mixt_i = \
                perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_augmented,
                                 y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                                 X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented,
                                 X_mixtures, target_classes)

            lrs_before_calib[key] = lrs_before_calib_i
            lrs_after_calib[key] = lrs_after_calib_i
            lrs_test_as_mixtures_before_calib[key] = lrs_test_as_mixtures_before_calib_i
            lrs_test_as_mixtures_after_calib[key] = lrs_test_as_mixtures_after_calib_i
            lrs_before_calib_mixt[key] = lrs_before_calib_mixt_i
            lrs_after_calib_mixt[key] = lrs_after_calib_mixt_i

    return lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib,\
           lrs_before_calib_mixt, lrs_after_calib_mixt


def plot_scatterplot_lrs(lrs, savefig=None, show=None):

    for method, data in lrs.items():
        fig, (axs1, axs2, axs3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
        plot_scatterplot_lr(data.lrs_after_calib, axs1)
        plot_scatterplot_lr(data.lrs_test_as_mixtures_after_calib, axs2)
        plot_scatterplot_lr(data.lrs_after_calib_mixt, axs3)

        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_' + method)
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()


def plot_scatterplot_lr(lrs, ax=None):
    ax = ax

    min_vals = []
    max_vals = []
    loglrs = OrderedDict()

    for prior, lr in lrs.items():
        loglrs[prior] = np.log10(lr)
        min_vals.append(np.min(np.log10(lr)))
        max_vals.append(np.max(np.log10(lr)))
    diagonal_coordinates = np.linspace(min(min_vals), max(max_vals))
    priors = list(lrs.keys())

    ax.scatter(loglrs[priors[0]], loglrs[priors[1]], s=3)
    ax.plot(diagonal_coordinates, diagonal_coordinates, 'k--', linewidth=1)
    ax.set_xlim(min(min_vals), max(max_vals))
    ax.set_ylim(min(min_vals), max(max_vals))
    ax.set_xlabel("log10(LR) {}".format(priors[0]))
    ax.set_ylabel("log10(LR) {}".format(priors[1]))

    return ax


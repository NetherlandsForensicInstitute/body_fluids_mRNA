"""

"""

import os
import time

import numpy as np
import rna.settings as settings

from collections import OrderedDict

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

        for i, binarize in enumerate(settings.binarize):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)

            augmented_data = OrderedDict()
            for p, priors in enumerate(settings.priors):

                # ======= Augment data =======
                augmented_data[str(priors)] = augment_splitted_data(X_calib, X_test, X_train, binarize, from_penile,
                                                                    label_encoder, n_celltypes, n_features, priors,
                                                                    y_calib, y_nhot_mixtures, y_test, y_train, AugmentedData)

            # ======= Convert data accordingly =======
            if binarize:
                X_train_transformed = np.where(combine_samples(X_train) > 150, 1, 0)
                X_calib_transformed = np.where(combine_samples(X_calib) > 150, 1, 0)
                X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            else:
                X_train_transformed = combine_samples(X_train) / 1000
                X_calib_transformed = combine_samples(X_calib) / 1000
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                try:
                    for k, models in enumerate(settings.models):
                        print("Fold {} \n Binarize the data: {} \n Use softmax to calculate probabilities with: {} \n Model: {}".format(n, binarize, softmax, models[0]))

                        # ======= Calculate LRs before and after calibration =======
                        if settings.augment:
                            lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, \
                            lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(
                                augmented_data, X_mixtures, n, binarize, softmax, models, mle, label_encoder, target_classes)




                        ### UNTIL HERE
                        else:
                            y_train_transformed = mle.inv_transform_single(y_train)
                            y_train_transformed = mle.labels_to_nhot(y_train_transformed)
                            y_calib_transformed = mle.inv_transform_single(y_calib)
                            y_calib_transformed = mle.labels_to_nhot(y_calib_transformed)
                            model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                            lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                                perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_transformed,
                                                 y_train_transformed, X_calib_transformed, y_calib_transformed, X_test_augmented,
                                                 y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes, save_hist=True)
                except:
                    # this is when the probabilities are calculated with the softmax function and the original data
                    # is used to train the models with.
                    continue
        end = time.time()
        print("Execution time in seconds: {}".format(end - start))


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

    lrs_before_calib = OrderedDict()
    for i, (keys1, data1) in enumerate(augmented_data.items()):
        print(keys1)

        lrs_tested_on_one_prior = []
        for j, (keys2, data2) in enumerate(augmented_data.items()):
            print(keys2)

        ## HERE
            model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, \
            lrs_before_calib_mixt, lrs_after_calib_mixt = \
                perform_analysis(n, binarize, softmax, models, mle, label_encoder, data2.X_train_augmented,
                                 data2.y_train_nhot_augmented, data2.X_calib_augmented, data2.y_calib_nhot_augmented,
                                 data1.X_test_augmented, data1.y_test_nhot_augmented, data1.X_test_as_mixtures_augmented,
                                 X_mixtures, target_classes)

            lrs_tested_on_one_prior.append(lrs_before_calib)

        lrs_before_calib[keys1] = lrs_tested_on_one_prior


    return lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, \
            lrs_before_calib_mixt, lrs_after_calib_mixt
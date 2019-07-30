"""
Run nfold analysis.
"""

import os
import time

import numpy as np
import rna.settings as settings

from sklearn.model_selection import train_test_split

from rna.analytics import cllr, combine_samples, calculate_accuracy_all_target_classes
from rna.augment import augment_data, MultiLabelEncoder, only_use_same_combinations_as_in_mixtures
from rna.lr_system import perform_analysis
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import vec2string, string2vec
from rna.plotting import plot_boxplot_of_metric


def nfold_analysis(nfolds, tc):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    # ======= Initialize =======
    emtpy_numpy_array = np.zeros((nfolds, len(settings.binarize), len(settings.softmax), len(settings.models)))
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


    for n in range(nfolds):
        print("Fold {}".format(n))

        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        for i, binarize in enumerate(settings.binarize):
            print(" Binarize the data: {} {}".format(binarize, i))
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)

            # ======= Augment data =======
            X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features,
                                                                     settings.nsamples[0], label_encoder, settings.priors,
                                                                     binarize=binarize, from_penile=from_penile)
            X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features,
                                                                     settings.nsamples[1], label_encoder, settings.priors,
                                                                     binarize=binarize, from_penile=from_penile)
            X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features,
                                                                   settings.nsamples[2], label_encoder, settings.priors,
                                                                   binarize=binarize, from_penile=from_penile)
            X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented = only_use_same_combinations_as_in_mixtures(X_test_augmented, y_test_nhot_augmented, y_nhot_mixtures)

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
                        model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                        lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                            perform_analysis(softmax, models, mle, X_train_augmented, y_train_nhot_augmented,
                                             X_calib_augmented, y_calib_nhot_augmented, X_test_augmented,
                                             y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures,
                                             target_classes)
                    else:
                        y_train_transformed = mle.inv_transform_single(y_train)
                        y_train_transformed = mle.labels_to_nhot(y_train_transformed)
                        y_calib_transformed = mle.inv_transform_single(y_calib)
                        y_calib_transformed = mle.labels_to_nhot(y_calib_transformed)
                        model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                        lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                            perform_analysis(softmax, models, mle, X_train_transformed, y_train_transformed,
                                             X_calib_transformed, y_calib_transformed, X_test_augmented,
                                             y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures,
                                             target_classes)


                    # ======= Calculate accuracy and log-likelihood-ratio cost =======
                    for t, target_class in enumerate(target_classes):
                        target_class_str = vec2string(target_class, label_encoder)

                        if settings.augment:
                            accuracies_train[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, y_train_nhot_augmented, X_train_augmented, target_classes)[t]
                        else:
                            accuracies_train[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, y_train_transformed, X_train_transformed, target_classes)[t]
                        accuracies_test[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)[t]
                        accuracies_test_as_mixtures[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, y_test_as_mixtures_nhot_augmented, X_test_as_mixtures_augmented, target_classes)[t]
                        accuracies_mixtures[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, y_nhot_mixtures, X_mixtures, target_classes)[t]
                        accuracies_single[target_class_str][n, i, j, k] = calculate_accuracy_all_target_classes(model, mle, mle.inv_transform_single(y_test), X_test_transformed, target_classes)[t]

                        cllr_test[target_class_str][n, i, j, k] = cllr(lrs_after_calib[:, t], y_test_nhot_augmented, target_class)
                        cllr_test_as_mixtures[target_class_str][n, i, j, k] = cllr(lrs_test_as_mixtures_after_calib[:, t], y_test_as_mixtures_nhot_augmented, target_class)
                        cllr_mixtures[target_class_str][n, i, j, k] = cllr(lrs_after_calib_mixt[:, t], y_nhot_mixtures, target_class)


    # ======= Visualize the performance metrics =======

    # for target_class in target_classes:
    #     target_class_str = vec2string(target_class, label_encoder)
    #
    #     plot_boxplot_of_metric(accuracies_train[target_class_str], "train accuracy")
    #     plot_boxplot_of_metric(accuracies_test[target_class_str], "test accuracy")
    #     plot_boxplot_of_metric(accuracies_test_as_mixtures[target_class_str], "test as mixt accuracy")
    #     plot_boxplot_of_metric(accuracies_mixtures[target_class_str], "mixtures accuracy")
    #     plot_boxplot_of_metric(accuracies_single[target_class_str], "singles accuracy")
    #
    #     plot_boxplot_of_metric(cllr_test[target_class_str], "test Cllr")
    #     plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], "test as mixt Cllr")
    #     plot_boxplot_of_metric(cllr_mixtures[target_class_str], "mixtures Cllr")


    # ======= Save the performance metrics =======
    for target_class in target_classes:
        target_class_str = vec2string(target_class, label_encoder)

        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plot_boxplot_of_metric(accuracies_train[target_class_str], "train accuracy",
                               savefig=os.path.join('scratch', 'boxplot_train_accuracy_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_test[target_class_str], "test accuracy",
                               savefig=os.path.join('scratch', 'boxplot_test_accuracy_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_test_as_mixtures[target_class_str], "test as mixt accuracy",
                               savefig=os.path.join('scratch', 'boxplot_test_as_mixtures_accuracy_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_mixtures[target_class_str], "mixtures accuracy",
                               savefig=os.path.join('scratch', 'boxplot_mixture_accuracy_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_single[target_class_str], "singles accuracy",
                               savefig=os.path.join('scratch', 'boxplot_single_accuracy_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_test[target_class_str], "test Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], "test as mixt Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_as_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(cllr_mixtures[target_class_str], "mixtures Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
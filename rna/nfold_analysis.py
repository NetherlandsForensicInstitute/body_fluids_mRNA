"""
Run nfold analysis.
"""

import os
import time

import numpy as np
import rna.settings as settings

from sklearn.model_selection import train_test_split

from rna.analytics import augment_data, cllr, combine_samples, perform_analysis, get_accuracy
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, MarginalXGBClassifier
from rna.utils import vec2string, string2vec, MultiLabelEncoder, only_use_same_combinations_as_in_mixtures
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
    accuracies, cllr_test, cllr_test_as_mixtures, cllr_mixtures = [dict() for i in range(4)]

    accuracies['train'] = emtpy_numpy_array.copy()
    accuracies['test'] = emtpy_numpy_array.copy()
    accuracies['test as mixtures'] = emtpy_numpy_array.copy()
    accuracies['mixture'] = emtpy_numpy_array.copy()
    accuracies['single'] = emtpy_numpy_array.copy()

    for target_class in target_classes:
        target_class_str = vec2string(target_class, label_encoder)
        cllr_test[target_class_str] = emtpy_numpy_array.copy()
        cllr_test_as_mixtures[target_class_str] = emtpy_numpy_array.copy()
        cllr_mixtures[target_class_str] = emtpy_numpy_array.copy()


    for n in range(nfolds):
        start = time.time()

        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        for i, binarize in enumerate(settings.binarize):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)

            # ======= Augment data =======
            X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features, settings.nsamples[0], label_encoder, binarize=binarize, from_penile=from_penile)
            X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features, settings.nsamples[1], label_encoder, binarize=binarize, from_penile=from_penile)
            X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features, settings.nsamples[2], label_encoder, binarize=binarize, from_penile=from_penile)
            X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented = only_use_same_combinations_as_in_mixtures(X_test_augmented, y_test_nhot_augmented, y_nhot_mixtures)

            # ======= Convert data accordingly =======
            if binarize:
                X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            else:
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                for k, models in enumerate(settings.models):
                    print("Fold {} \n Binarize the data: {} \n Use softmax to calculate probabilities with: {} \n Model: {}".format(n, binarize, softmax, models[0]))

                    # ======= Calculate LRs before and after calibration =======
                    model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                    lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                        perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_augmented,
                                         y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                                         X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented,
                                         X_mixtures, target_classes, )

                    # ======= Calculate accuracy =======
                    accuracies['train'][n, i, j, k] = get_accuracy(model, mle, y_train_nhot_augmented, X_train_augmented, target_classes)
                    accuracies['test'][n, i, j, k] = get_accuracy(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)
                    accuracies['test as mixtures'][n, i, j, k] = get_accuracy(model, mle, y_test_as_mixtures_nhot_augmented, X_test_as_mixtures_augmented, target_classes)
                    accuracies['mixture'][n, i, j, k] = get_accuracy(model, mle, y_nhot_mixtures, X_mixtures, target_classes)
                    accuracies['single'][n, i, j, k] = get_accuracy(model, mle, mle.inv_transform_single(y_test), X_test_transformed, target_classes)

                    # ======= Calculate log-likelihood-ratio cost =======
                    for t, target_class in enumerate(target_classes):
                        target_class_str = vec2string(target_class, label_encoder)
                        cllr_test[target_class_str][n, i, j, k] = cllr(lrs_after_calib[:, t], y_test_nhot_augmented, target_class)
                        cllr_test_as_mixtures[target_class_str][n, i, j, k] = cllr(lrs_test_as_mixtures_after_calib[:, t], y_test_as_mixtures_nhot_augmented, target_class)
                        cllr_mixtures[target_class_str][n, i, j, k] = cllr(lrs_after_calib_mixt[:, t], y_nhot_mixtures, target_class)
        end = time.time()
        print("Execution time in seconds: {}".format(end - start))

    plot_boxplot_of_metric(accuracies['train'], "train accuracy", savefig=os.path.join('scratch', 'boxplot_train_accuracy'))
    plot_boxplot_of_metric(accuracies['test'], "test accuracy", savefig=os.path.join('scratch', 'boxplot_test_accuracy'))
    plot_boxplot_of_metric(accuracies['test as mixtures'], "test as mixtures", savefig=os.path.join('scratch', 'boxplot_test_as_mixtures'))
    plot_boxplot_of_metric(accuracies['mixture'], "mixture accuracy", savefig=os.path.join('scratch', 'boxplot_mixture_accuracy'))
    plot_boxplot_of_metric(accuracies['single'], "single accuracy", savefig=os.path.join('scratch', 'boxplot_single_accuracy'))

    for target_class in target_classes:
        target_class_str = vec2string(target_class, label_encoder)

        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plot_boxplot_of_metric(cllr_test[target_class_str], "cllr test {}".format(target_class_str),
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], "cllr test as mixtures {}".format(target_class_str),
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_as_mixtures_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_mixtures[target_class_str], "cllr mixtures {}".format(target_class_str),
                               savefig=os.path.join('scratch', 'boxplot_cllr_mixtures_{}'.format(target_class_save)))

    print("END")


def model_with_correct_settings(model_no_settings, softmax):
    """
    Ensures that the correct model with correct settings is used in the analysis.
    This is based on a string 'model_no_settings' and a boolean deciding how the
    probabilties are calculated 'probabilitye_calculation': either with the softmax
    function or the sigmoid function.

    :param model_no_settings: str: model
    :param probability_calculation: boolean: if True the softmax function is used to
        calculate the probabilities with.
    :return: model with correct settings
    """

    if model_no_settings == 'MLP':
        if softmax:
            model = MarginalMLPClassifier()
        else:
            model = MarginalMLPClassifier(activation='logistic')

    elif model_no_settings == 'MLR':
        if softmax:
            model = MarginalMLRClassifier(multi_class='multinomial', solver='newton-cg')
        else:
            model = MarginalMLRClassifier()

    elif model_no_settings == 'XGB':
        if softmax:
            model = MarginalXGBClassifier()
        else:
            model = MarginalXGBClassifier(method='sigmoid')

    else:
        raise ValueError("No class exists for this model")

    return model
"""
Performs project specific.
"""

import os
from collections import OrderedDict

# import keras
import numpy as np
from lir import calculate_cllr
from sklearn.metrics import accuracy_score
from typing import List

from rna.constants import nhot_matrix_all_combinations, DEBUG
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, \
    MarginalXGBClassifier, MarginalDLClassifier, \
    MarginalRFClassifier, MarginalSVMClassifier
from rna.plotting import plot_calibration_process, plot_insights_cllr, plot_coefficient_importances


def combine_samples(data_for_class: List):
    """
    Combines the repeated measurements per sample.

    :param data_for_class: List of  N_observations_per_sample x N_markers measurements numpy array, where number of observations may differ per sample
    :return: N_samples x N_markers measurements numpy array
    """
    data_for_class_mean = np.array([np.mean(data_for_class[i], axis=0)
                                    for i in range(len(data_for_class))])
    return data_for_class_mean


def generate_lrs(X_train, y_train, X_calib, y_calib, X_test, X_test_as_mixtures, X_mixtures, target_classes, model, mle,
                 softmax, calibration_on_loglrs, do_calibration):
    """
    When softmax the model must be fitted on labels, whereas with sigmoid the model must be fitted on
    an nhot encoded vector representing the labels. Ensure that labels take the correct form, fit the
    model and predict the lrs before and after calibration for both X_test and X_mixtures.
    """

    if softmax:  # y_train must be list with labels
        try:
            y_train = mle.nhot_to_labels(y_train)
            # y_test = mle.nhot_to_labels(y_test)
        except:  # already are labels
            pass
        try:
            # for DL model y_train must always be nhot encoded
            # TODO: Find better solution
            if isinstance(model._classifier, keras.engine.training.Model):
                y_train = np.eye(2 ** 8)[y_train]
        except:
            pass
    else:  # y_train must be nhot encoded labels
        try:
            y_train = mle.labels_to_nhot(y_train)
        except:  # already is nhot encoded
            pass
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(y_train[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    try:  # y_calib must always be nhot encoded
        y_calib = mle.labels_to_nhot(y_calib)
    except:  # already is nhot encoded
        pass

    model.fit_classifier(X_train, y_train)
    if do_calibration:
        model.fit_calibration(X_calib, y_calib, target_classes, calibration_on_loglrs=calibration_on_loglrs)

    lrs_before_calib = model.predict_lrs(X_test, target_classes, with_calibration=False)
    if do_calibration:
        lrs_after_calib = model.predict_lrs(X_test, target_classes, calibration_on_loglrs=calibration_on_loglrs)
    else:
        lrs_after_calib = lrs_before_calib

    try:
        lrs_before_calib_test_as_mixtures = model.predict_lrs(X_test_as_mixtures, target_classes,
                                                              with_calibration=False)
        if do_calibration:
            lrs_after_calib_test_as_mixtures = model.predict_lrs(X_test_as_mixtures, target_classes,
                                                                 calibration_on_loglrs=calibration_on_loglrs)
        else:
            lrs_after_calib_test_as_mixtures = lrs_before_calib_test_as_mixtures
    except TypeError:
        # When there are no samples from the synthetic data with the same labels as in the original mixtures data.
        lrs_before_calib_test_as_mixtures = np.zeros([1, lrs_before_calib.shape[1]])
        lrs_after_calib_test_as_mixtures = np.zeros([1, lrs_before_calib.shape[1]])

    lrs_before_calib_mixt = model.predict_lrs(X_mixtures, target_classes, with_calibration=False)
    if do_calibration:
        lrs_after_calib_mixt = model.predict_lrs(X_mixtures, target_classes,
                                                 calibration_on_loglrs=calibration_on_loglrs)
    else:
        lrs_after_calib_mixt = lrs_before_calib_mixt

    return model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, \
           lrs_before_calib_mixt, lrs_after_calib_mixt


def clf_with_correct_settings(clf_no_settings, softmax: bool, n_classes: int, with_calibration: bool):
    """
    Ensures that the correct classifier with correct settings is used in the analysis. This is based on a string
    'model_no_settings' and a boolean deciding how the probabilties are calculated 'softmax': either with the softmax
    function or the sigmoid function.

    :param clf_no_settings: str: classifier
    :param softmax: bool: whether probabilities are calculated with softmax
    :param n_classes: int: number of classes
    :return: classifier with correct settings
    """
    assert type(softmax) == bool
    if clf_no_settings == 'MLP':
        # TODO this does not appear to be what we want it to be?
        if softmax:
            classifier = MarginalMLPClassifier()
        else:
            classifier = MarginalMLPClassifier(activation='logistic')

    elif clf_no_settings == 'SVM':
        if softmax:
            classifier = MarginalSVMClassifier()
        else:
            classifier = MarginalSVMClassifier(multi_label='ovr')

    elif clf_no_settings == 'MLR':
        if softmax:
            classifier = MarginalMLRClassifier(multi_class='multinomial', solver='newton-cg')
        else:
            classifier = MarginalMLRClassifier(multi_class='ovr')

    elif clf_no_settings == 'XGB':
        if softmax:
            classifier = MarginalXGBClassifier()
        else:
            classifier = MarginalXGBClassifier(method='sigmoid')

    elif clf_no_settings == 'DL':
        if softmax:
            classifier = MarginalDLClassifier(n_classes=2 ** 8, activation_layer='softmax',
                                              optimizer="adam", loss="categorical_crossentropy", epochs=30)
        else:
            classifier = MarginalDLClassifier(n_classes=n_classes, activation_layer='sigmoid',
                                              optimizer="adam", loss="binary_crossentropy", epochs=30)
    elif clf_no_settings == 'RF':
        if softmax:
            classifier = MarginalRFClassifier()
        else:
            classifier = MarginalRFClassifier(multi_label='ovr')

    else:
        raise ValueError("No class exists for this classifier: {}".format(clf_no_settings))

    return classifier


def perform_analysis(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                     X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes,
                     present_markers, models, mle, label_encoder, method_name_prior, softmax, calibration_on_loglrs,
                     output_folder=None):
    """
    Selects the model with correct settings with 'model' and 'softmax' and calculates the likelihood-ratio's before and
    after calibration on three test sets (augmented test, original mixtures and augmented test as mixtures).

    :param output_folder:
    :param present_markers:
    :param method_name_prior: str: model and settings to save plots with
    :param calibration_on_loglrs: bool: whether calibration is fitted on loglrs otherwise on probability
    :param output_folder: specify if you want plots (will be in subfolder plots). Otherwise leave None
    """

    classifier = models[0]
    with_calibration = models[1]

    model = clf_with_correct_settings(classifier, softmax, n_classes=target_classes.shape[0], with_calibration=with_calibration)
    if not with_calibration:
        # hacky, nicer to do at instantiation
        model._calibrator=None

    if with_calibration:  # with calibration
        model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, \
        lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                         X_test_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes, model, mle,
                         softmax, calibration_on_loglrs, do_calibration=with_calibration)

        if output_folder and DEBUG:
            # calibration data
            plot_calibration_process(model.predict_lrs(X_calib_augmented, target_classes, with_calibration=False),
                                     y_calib_nhot_augmented, model._calibrators_per_target_class, None, target_classes,
                                     label_encoder, calibration_on_loglrs,
                                     savefig=os.path.join(output_folder, 'plots',
                                                          'calib_process_calib_{}'.format(method_name_prior)))

            # test data
            plot_calibration_process(model.predict_lrs(X_test_augmented, target_classes, with_calibration=False),
                                     y_test_nhot_augmented, model._calibrators_per_target_class,
                                     (lrs_before_calib, lrs_after_calib), target_classes, label_encoder,
                                     calibration_on_loglrs,
                                     savefig=os.path.join(output_folder, 'plots',
                                                          'calib_process_test_{}'.format(method_name_prior)))

    else:  # no calibration
        X_train = np.concatenate((X_train_augmented, X_calib_augmented), axis=0)
        y_train = np.concatenate((y_train_nhot_augmented, y_calib_nhot_augmented), axis=0)
        X_calib = np.array([])
        y_calib = np.array([])

        model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, \
        lrs_before_calib_mixt, lrs_after_calib_mixt = generate_lrs(X_train, y_train, X_calib, y_calib, X_test_augmented,
                                                                   X_test_as_mixtures_augmented, X_mixtures,
                                                                   target_classes, model, mle, softmax,
                                                                   calibration_on_loglrs,
                                                                   do_calibration=with_calibration)

        assert np.array_equal(lrs_before_calib, lrs_after_calib), \
            "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(
                with_calibration)
        assert np.array_equal(lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures), \
            "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(
                with_calibration)
        assert np.array_equal(lrs_before_calib_mixt, lrs_after_calib_mixt), \
            "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(
                with_calibration)

        # bootstrap LRs
        # B = 1
        # all_lrs_after_calib_bs = np.zeros([lrs_after_calib.shape[0], lrs_after_calib.shape[1], B])
        # for b in range(B):
        #     # throw away random 20% from train data
        #     # sample_indices = np.random.choice(np.arange(X_train.shape[0]), size=int(0.8 * X_train.shape[0]), replace=False)
        #
        #     # sample with replacement
        #     # TODO: Take into account equal size for h1 and h2
        #     sample_indices = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
        #
        #     X_train_bs = X_train[sample_indices, :]
        #     y_train_bs = y_train[sample_indices, :]
        #
        #     _, _, lrs_after_calib_bs, _, _, _, _ = generate_lrs(X_train_bs, y_train_bs, X_calib, y_calib,
        #                                                         X_test_augmented, y_test_nhot_augmented,
        #                                                         X_test_as_mixtures_augmented, X_mixtures,
        #                                                         target_classes, model, model_tc, mle, softmax,
        #                                                         calibration_on_loglrs)
        #     all_lrs_after_calib_bs[:, :, b] = lrs_after_calib_bs
        #
        # # TODO: Think what to do with upper and lower bounds
        # lower_bounds_tc, upper_bounds_tc = plot_lrs_with_bootstrap_ci(lrs_after_calib, all_lrs_after_calib_bs,
        #                                                               target_classes, label_encoder)

    # Plot the values of the coefficients to see if MLR uses the correct features (markers).
    if output_folder:
        try:
            if DEBUG:
                if classifier == 'MLR':
                    plot_coefficient_importances(model, target_classes, present_markers, label_encoder,
                                                 savefig=os.path.join(output_folder, 'plots',
                                                                      'coefficient_importance_{}'.format(
                                                                          method_name_prior)))

                plot_insights_cllr(lrs_after_calib,
                                        y_test_nhot_augmented, target_classes, label_encoder,
                               savefig=os.path.join(output_folder, 'plots',
                                                    'insights_cllr_calculation_{}'.format(method_name_prior)))
        except:
            pass

    return model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, \
           lrs_after_calib_test_as_mixtures, lrs_before_calib_mixt, lrs_after_calib_mixt


def calculate_lrs_for_different_priors(augmented_data, X_mixtures, target_classes, baseline_prior, present_markers,
                                       models, mle, label_encoder, method_name, softmax, calibration_on_loglrs,
                                       save_path):
    """
    Calculates the likelihood-ratio's before and after calibration for all priors. The after_adjusting_dl prior is used to
    select the test data (i.e. the data with which the likelihood ratio's are calculated) with. Returns for each test
    set a dictionary where keys are priors and the values contain the lr's.

    :param present_markers:
    :param baseline_prior: str: string of list that will enable selecting the correct test data
    :param method_name: str: model and settings to save plots with
    :param calibration_on_loglrs: bool: whether calibration is fitted on loglrs otherwise on probability
    """

    # must be tested on the same test data based on after_adjusting_dl prior
    X_test_augmented = augmented_data[baseline_prior].X_test_augmented
    y_test_nhot_augmented = augmented_data[baseline_prior].y_test_nhot_augmented
    X_test_as_mixtures_augmented = augmented_data[baseline_prior].X_test_as_mixtures_augmented
    y_test_as_mixtures_nhot_augmented = augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented

    model = OrderedDict()
    lrs_before_calib = OrderedDict()
    lrs_after_calib = OrderedDict()
    lrs_before_calib_test_as_mixtures = OrderedDict()
    lrs_after_calib_test_as_mixtures = OrderedDict()
    lrs_before_calib_mixt = OrderedDict()
    lrs_after_calib_mixt = OrderedDict()

    for key, data in augmented_data.items():
        method_name_prior = method_name + '_' + key

        X_train_augmented = data.X_train_augmented
        y_train_nhot_augmented = data.y_train_nhot_augmented
        X_calib_augmented = data.X_calib_augmented
        y_calib_nhot_augmented = data.y_calib_nhot_augmented

        model_i, lrs_before_calib_i, lrs_after_calib_i, \
        lrs_before_calib_test_as_mixtures_i, lrs_after_calib_test_as_mixtures_i, \
        lrs_before_calib_mixt_i, lrs_after_calib_mixt_i = \
            perform_analysis(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                             X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures,
                             target_classes, present_markers, models, mle, label_encoder, method_name_prior, softmax,
                             calibration_on_loglrs, output_folder=save_path)

        model[key] = model_i
        lrs_before_calib[key] = lrs_before_calib_i
        lrs_after_calib[key] = lrs_after_calib_i
        lrs_before_calib_test_as_mixtures[key] = lrs_before_calib_test_as_mixtures_i
        lrs_after_calib_test_as_mixtures[key] = lrs_after_calib_test_as_mixtures_i
        lrs_before_calib_mixt[key] = lrs_before_calib_mixt_i
        lrs_after_calib_mixt[key] = lrs_after_calib_mixt_i

    return model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
           lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
           lrs_before_calib_mixt, lrs_after_calib_mixt


def calculate_accuracy_all_target_classes(X, y_true, target_classes, model, mle):
    """
    Predicts labels and ensures that both the true and predicted labels are nhot encoded. Calculates the accuracy for
    all target classes and stores it in a list. The set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    :return: accuracy_scores: list with accuracy for all target classes
    """

    y_pred = model._classifier.predict(X)
    try:
        if isinstance(model._classifier, keras.engine.training.Model):
            # when the model predicts probabilities rather than the binary classes
            # this is only the case for the DL model
            if y_pred.shape[1] == 2 ** 8:
                unique_vectors = np.flip(np.unique(nhot_matrix_all_combinations, axis=0), axis=1)
                y_pred = np.array([np.sum(y_pred[:, np.argwhere(unique_vectors[:, i] == 1).flatten()], axis=1) for i in
                                   range(unique_vectors.shape[1])]).T
                y_pred = np.where(y_pred > 0.5, 1, 0)
            else:
                y_pred = np.where(y_pred > 0.5, 1, 0)
    except:
        pass

    try:
        y_true = mle.labels_to_nhot(y_true)
    except:
        pass

    try:
        y_pred = mle.labels_to_nhot(y_pred)
    except:
        pass

    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(len(y_pred), 1)

    indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
    y_true = np.array([np.max(np.array(y_true[:, indices[i]]), axis=1) for i in range(len(indices))]).T
    if y_pred.shape[1] != len(target_classes):
        y_pred = np.array([np.max(np.array(y_pred[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    accuracy_scores = []
    for t, target_class in enumerate(target_classes):
        accuracy_scores.append(accuracy_score(y_true[:, t], y_pred[:, t]))

    return accuracy_scores


def cllr(lrs, y_nhot, target_class):
    """
    Computes the Cllr (log-likelihood ratio cost) for one target class.

    :param lrs: numpy array: N_samples with the LRs from the method
    :param y_nhot: N_samples x N_single_cell_type n_hot encoding of the labels
    :param target_class: vector of length n_single_cell_types with at least one 1
    :return: float: the log-likehood ratio cost
    """

    lrs1 = lrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1)].flatten()
    lrs2 = lrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0)].flatten()

    if len(lrs1) > 0 and len(lrs2) > 0:
        return calculate_cllr(lrs2, lrs1).cllr
    else:
        # no ground truth labels for the celltype, so cannot calculate the cllr.
        return 9999.0000


def append_lrs_for_all_folds(lrs_for_model, type):
    """
    Concatenates the lrs calculated on test data for each fold.

    :param lrs_for_model:
    :param type: str: the test data for which lrs should be concatenated.
    """

    lrs_before_for_all_methods = OrderedDict()
    lrs_after_for_all_methods = OrderedDict()
    y_nhot_for_all_methods = OrderedDict()
    for fold, methods in lrs_for_model.items():

        for method, data in methods.items():
            priors = list(data.lrs_after_calib.keys())

            for prior in priors:
                prior_method = f'{method}_{prior}'

                if prior_method in lrs_after_for_all_methods:
                    if type == 'test augm':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method],
                                                                             data.lrs_before_calib[prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method],
                                                                            data.lrs_after_calib[prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method],
                                                                         data.y_test_nhot_augmented, axis=0)
                    elif type == 'test augm as mixt':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method],
                                                                             data.lrs_before_calib_test_as_mixtures[
                                                                                 prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method],
                                                                            data.lrs_after_calib_test_as_mixtures[
                                                                                prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method],
                                                                         data.y_test_as_mixtures_nhot_augmented, axis=0)
                    elif type == 'mixt':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method],
                                                                             data.lrs_before_calib_mixt[prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method],
                                                                            data.lrs_after_calib_mixt[prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method],
                                                                         data.y_mixtures_nhot, axis=0)
                else:
                    if type == 'test augm':
                        lrs_before_for_all_methods[prior_method] = data.lrs_before_calib[prior]
                        lrs_after_for_all_methods[prior_method] = data.lrs_after_calib[prior]
                        y_nhot_for_all_methods[prior_method] = data.y_test_nhot_augmented
                    elif type == 'test augm as mixt':
                        lrs_before_for_all_methods[prior_method] = data.lrs_before_calib_test_as_mixtures[prior]
                        lrs_after_for_all_methods[prior_method] = data.lrs_after_calib_test_as_mixtures[prior]
                        y_nhot_for_all_methods[prior_method] = data.y_test_as_mixtures_nhot_augmented
                    elif type == 'mixt':
                        lrs_before_for_all_methods[prior_method] = data.lrs_before_calib_mixt[prior]
                        lrs_after_for_all_methods[prior_method] = data.lrs_after_calib_mixt[prior]
                        y_nhot_for_all_methods[prior_method] = data.y_mixtures_nhot

    return lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods

# TODO: Check if want to keep
# def use_repeated_measurements_as_single(X_single, y_nhot_single, y_single):
#     """
#     Treats each repeated measurement as an individual sample and transforms the
#     original data sets accordingly.
#     """
#
#     N = X_single.size
#     X_single_nrp = []
#     y_nhot_single_nrp = []
#     y_single_nrp = []
#     for i in range(N):
#         n = X_single[i].shape[0]
#         y_nhot_single_i = np.tile(y_nhot_single[i, :], (n, 1))
#         y_single_nrp.extend(y_single[i].tolist() * n)
#         for j in range(n):
#             X_single_nrp.append(X_single[i][j])
#             y_nhot_single_nrp.append(y_nhot_single_i[j, :])
#
#     X_single_nrp = np.asarray(X_single_nrp)
#     y_nhot_single_nrp = np.asarray(y_nhot_single_nrp)
#     y_single_nrp = np.asarray(y_single_nrp)
#
#     return X_single_nrp, y_nhot_single_nrp, y_single_nrp

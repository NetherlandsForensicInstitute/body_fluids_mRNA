"""
Performs project specific.
"""
from collections import OrderedDict

import keras
import numpy as np
from sklearn.metrics import accuracy_score

from rna.constants import nhot_matrix_all_combinations

from lir.lr import calculate_cllr
from lir.plotting import makeplot_hist_density, plot_scatterplot_lr_before_after_calib
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, MarginalXGBClassifier, MarginalDLClassifier
# from rna.plotting import plot_scatterplot_lr_before_after_calib


def combine_samples(data_for_class):
    """
    Combines the repeated measurements per sample.

    :param data_for_class: N_samples x N_observations_per_sample x N_markers measurements numpy array
    :return: N_samples x N_markers measurements numpy array
    """
    data_for_class_mean = np.array([np.mean(data_for_class[i], axis=0)
                                    for i in range(data_for_class.shape[0])])
    return data_for_class_mean


def use_repeated_measurements_as_single(X_single, y_nhot_single, y_single):
    """
    Treats each repeated measurement as an individual sample and transforms the
    original data sets accordingly.
    """

    N = X_single.size
    X_single_nrp = []
    y_nhot_single_nrp = []
    y_single_nrp = []
    for i in range(N):
        n = X_single[i].shape[0]
        y_nhot_single_i = np.tile(y_nhot_single[i, :], (n, 1))
        y_single_nrp.extend(y_single[i].tolist() * n)
        for j in range(n):
            X_single_nrp.append(X_single[i][j])
            y_nhot_single_nrp.append(y_nhot_single_i[j, :])

    X_single_nrp = np.asarray(X_single_nrp)
    y_nhot_single_nrp = np.asarray(y_nhot_single_nrp)
    y_single_nrp = np.asarray(y_single_nrp)

    return X_single_nrp, y_nhot_single_nrp, y_single_nrp


def generate_lrs(X_train, y_train, X_calib, y_calib, X_test, y_test, X_test_as_mixtures, X_mixtures, model,
                 target_classes, mle, softmax, calibration_on_loglrs):
    """
    When softmax the model must be fitted on labels, whereas with sigmoid the model must be fitted on
    an nhot encoded vector representing the labels. Ensure that labels take the correct form, fit the
    model and predict the lrs before and after calibration for both X_test and X_mixtures.
    :param calibration_on_loglrs:
    """

    if softmax: # y_train must be list with labels
        try:
            y_train = mle.nhot_to_labels(y_train)
            y_test = mle.nhot_to_labels(y_test)
        except: # already are labels
            pass
        # for DL model y_train must always be nhot encoded
        # TODO: Find better solution
        if isinstance(model._classifier, keras.engine.training.Model):
            y_train = np.eye(2 ** 8)[y_train]
            y_test = np.eye(2 ** 8)[y_test]
    else: # y_train must be nhot encoded labels
        try:
            y_train = mle.labels_to_nhot(y_train)
            y_test = mle.labels_to_nhot(y_test)
        except: # already is nhot encoded
            pass
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(y_train[:, indices[i]]), axis=1) for i in range(len(indices))]).T
        y_test = np.array([np.max(np.array(y_test[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    try: # y_calib must always be nhot encoded
        y_calib = mle.labels_to_nhot(y_calib)
    except: # already is nhot encoded
        pass

    ## TO TEST DL --> CAN BE REMOVED LATER ON. Should then also remove y_test !
    # test_dl_model(model, X_train, y_train, X_test, y_test, target_classes)

    model.fit_classifier(X_train, y_train)
    model.fit_calibration(X_calib, y_calib, target_classes, calibration_on_loglrs=calibration_on_loglrs)

    lrs_before_calib = model.predict_lrs(X_test, target_classes, with_calibration=False)
    lrs_after_calib = model.predict_lrs(X_test, target_classes, calibration_on_loglrs=calibration_on_loglrs)

    lrs_test_as_mixtures_before_calib = model.predict_lrs(X_test_as_mixtures, target_classes, with_calibration=False)
    lrs_test_as_mixtures_after_calib = model.predict_lrs(X_test_as_mixtures, target_classes, calibration_on_loglrs=calibration_on_loglrs)

    lrs_before_calib_mixt = model.predict_lrs(X_mixtures, target_classes, with_calibration=False)
    lrs_after_calib_mixt = model.predict_lrs(X_mixtures, target_classes, calibration_on_loglrs=calibration_on_loglrs)

    return model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, \
           lrs_before_calib_mixt, lrs_after_calib_mixt


def cllr(lrs, y_nhot, target_class):
    """
    Computes the cllr for one celltype.

    :param lrs: numpy array: N_samples with the LRs from the method
    :param y_nhot: N_samples x N_single_cell_type n_hot encoding of the labels
    :param target_class: vector of length n_single_cell_types with at least one 1
    :return: float: the log-likehood cost ratio
    """
    lrs1 = lrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1)].flatten()
    lrs2 = lrs[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0)].flatten()

    if len(lrs1) > 0 and len(lrs2) > 0:
        return calculate_cllr(lrs2, lrs1).cllr
    else:
        # no ground truth labels for the celltype, so cannot calculate
        # the cllr.
        return 9999.0000


def calculate_accuracy_all_target_classes(model, mle, y_true, X, target_classes):
    """
    Predicts labels and ensures that both the true and predicted labels are nhot encoded.
    Calculates the accuracy.

    :return: accuracy: the set of labels predicted for a sample must *exactly* match the
        corresponding set of labels in y_true.
    """

    y_pred = model._classifier.predict(X)
    if isinstance(model._classifier, keras.engine.training.Model):
        # when the model predicts probabilities rather than the binary classes
        # this is only the case for the DL model
        if y_pred.shape[1] == 2 ** 8:
            unique_vectors = np.flip(np.unique(nhot_matrix_all_combinations, axis=0), axis=1)
            y_pred = np.array([np.sum(y_pred[:, np.argwhere(unique_vectors[:, i] == 1).flatten()], axis=1) for i in range(unique_vectors.shape[1])]).T
            y_pred = np.where(y_pred > 0.5, 1, 0)
        else:
            y_pred = np.where(y_pred > 0.5, 1, 0)

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


## TO TEST DL --> CAN BE REMOVED LATER ON
import matplotlib.pyplot as plt
def test_dl_model(model, X_train, y_train, X_test, y_test, target_classes):

    history = model._classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def calculate_lrs_for_different_priors(augmented_data, X_mixtures, target_classes, baseline_prior, models, mle,
                                       label_encoder, softmax, calibration_on_loglrs):

    # must be tested on the same test data based on baseline prior
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

    for i, (key, data) in enumerate(augmented_data.items()):
        print(" Prior: {}".format(key))

        X_train_augmented = data.X_train_augmented
        y_train_nhot_augmented = data.y_train_nhot_augmented
        X_calib_augmented = data.X_calib_augmented
        y_calib_nhot_augmented = data.y_calib_nhot_augmented

        model_i, lrs_before_calib_i, lrs_after_calib_i, \
        lrs_test_as_mixtures_before_calib_i, lrs_test_as_mixtures_after_calib_i, \
        lrs_before_calib_mixt_i, lrs_after_calib_mixt_i = \
            perform_analysis(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                             X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures,
                             target_classes, models, mle, label_encoder, softmax, calibration_on_loglrs, save_kde=True)

        model[key] = model_i
        lrs_before_calib[key] = lrs_before_calib_i
        lrs_after_calib[key] = lrs_after_calib_i
        lrs_before_calib_test_as_mixtures[key] = lrs_test_as_mixtures_before_calib_i
        lrs_after_calib_test_as_mixtures[key] = lrs_test_as_mixtures_after_calib_i
        lrs_before_calib_mixt[key] = lrs_before_calib_mixt_i
        lrs_after_calib_mixt[key] = lrs_after_calib_mixt_i

    return model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
           lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
           lrs_before_calib_mixt, lrs_after_calib_mixt


def perform_analysis(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                     X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes,
                     models, mle, label_encoder, softmax, calibration_on_loglrs, save_kde):

    classifier = models[0]
    with_calibration = models[1]

    model = model_with_correct_settings(classifier, softmax, n_classes=target_classes.shape[0])

    if with_calibration: # with calibration
        model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, \
        lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                         X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, model,
                         target_classes, mle, softmax, calibration_on_loglrs)

        # TODO: Check if want to keep
        if save_kde:
            makeplot_hist_density(model.predict_lrs(X_calib_augmented, target_classes, with_calibration=False),
                                  y_calib_nhot_augmented, model._calibrators_per_target_class, target_classes,
                                  label_encoder, calibration_on_loglrs)

            plot_scatterplot_lr_before_after_calib(lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder)

    else: # no calibration
        model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(np.concatenate((X_train_augmented, X_calib_augmented), axis=0),
                         np.concatenate((y_train_nhot_augmented, y_calib_nhot_augmented), axis=0), np.array([]),
                         np.array([]), X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented,
                         X_mixtures, model, target_classes, mle, softmax, calibration_on_loglrs)

        assert np.array_equal(lrs_before_calib, lrs_after_calib), "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(with_calibration)
        assert np.array_equal(lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures), "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(with_calibration)
        assert np.array_equal(lrs_before_calib_mixt, lrs_after_calib_mixt), "LRs before and after calibration are not the same, even though 'with calibration' is {}".format(with_calibration)

    return model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, \
           lrs_after_calib_test_as_mixtures, lrs_before_calib_mixt, lrs_after_calib_mixt


def model_with_correct_settings(model_no_settings, softmax, n_classes):
    """
    Ensures that the correct model with correct settings is used in the analysis.
    This is based on a string 'model_no_settings' and a boolean deciding how the
    probabilties are calculated 'softmax': either with the softmax
    function or the sigmoid function.

    :param n_classes:
    :param model_no_settings: str: model
    :param softmax: boolean: if True the softmax function is used to
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

    elif model_no_settings == 'DL':
        if softmax:
            model = MarginalDLClassifier(n_classes=2 ** 8, activation_layer='softmax',
                                         optimizer="adam", loss="categorical_crossentropy", epochs=150)
        else:
            model = MarginalDLClassifier(n_classes=n_classes, activation_layer='sigmoid',
                                         optimizer="adam", loss="binary_crossentropy", epochs=30)

    else:
        raise ValueError("No class exists for this model")

    return model


def combine_lrs_for_all_folds(lrs_for_model, type):
    """
    Combines the lrs calculated on test data for each fold.

    :param type:
    :param lrs_for_model:
    :return:
    """

    lrs_before_for_all_methods = OrderedDict()
    lrs_after_for_all_methods = OrderedDict()
    y_nhot_for_all_methods = OrderedDict()
    for fold, methods in lrs_for_model.items():

        for method, data in methods.items():
            priors = list(data.lrs_after_calib.keys())

            for prior in priors:
                prior_method = method + '_' + prior

                if prior_method in lrs_after_for_all_methods:
                    if type == 'test augm':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method], data.lrs_before_calib[prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method], data.lrs_after_calib[prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method], data.y_test_nhot_augmented, axis=0)
                    elif type == 'test augm as mixt':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method], data.lrs_before_calib_test_as_mixtures[prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method], data.lrs_after_calib_test_as_mixtures[prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method], data.y_test_as_mixtures_nhot_augmented, axis=0)
                    elif type == 'mixt':
                        lrs_before_for_all_methods[prior_method] = np.append(lrs_before_for_all_methods[prior_method], data.lrs_before_calib_mixt[prior], axis=0)
                        lrs_after_for_all_methods[prior_method] = np.append(lrs_after_for_all_methods[prior_method], data.lrs_after_calib_mixt[prior], axis=0)
                        y_nhot_for_all_methods[prior_method] = np.append(y_nhot_for_all_methods[prior_method], data.y_mixtures_nhot, axis=0)
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
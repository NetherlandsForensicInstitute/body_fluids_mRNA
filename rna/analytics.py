"""
Performs project specific.
"""
import keras
import numpy as np
from sklearn.metrics import accuracy_score

from rna.constants import single_cell_types, nhot_matrix_all_combinations

from lir.lr import calculate_cllr


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


def convert_prob_to_marginal_per_class(prob, target_classes, MAX_LR, priors_numerator=None, priors_denominator=None):
    """
    Converts n_samples x n_mixtures matrix of probabilities to a n_samples x n_target_classes
    matrix by summing over the probabilities containing the celltype(s) of interest.

    :param prob: n_samples x n_mixtures containing the predicted probabilities
    :param target_classes: n_target_classes x n_celltypes containing the n hot encoded classes of interest
    :param MAX_LR: int
    :param priors_numerator: vector of length n_single_cell_types, specifying 0 indicates we know this single cell type
    does not occur, specify 1 indicates we know this cell type certainly occurs, anything else assume implicit uniform
    distribution
    :param priors_denominator: vector of length n_single_cell_types, specifying 0 indicates we know this single cell type
    does not occur, specify 1 indicates we know this cell type certainly occurs, anything else assume implicit uniform
    distribution
    :return: n_samples x n_target_classes of probabilities
    """
    assert priors_numerator is None or type(priors_numerator) == list or type(priors_numerator) == np.ndarray
    assert priors_denominator is None or type(priors_denominator) == list or type(priors_denominator) == np.ndarray
    lrs = np.zeros((len(prob), len(target_classes)))
    for i, target_class in enumerate(target_classes):
        assert sum(target_class) > 0, 'No cell type given as target class'

        if prob.shape[1] == 2 ** target_classes.shape[1]: # lps
            # numerator
            indices_of_target_class = get_mixture_columns_for_class(target_class, priors_numerator)
            numerator = np.sum(prob[:, indices_of_target_class], axis=1)

            # denominator
            # TODO: Does this work when priors are defined?
            # TODO: Rewrite with priors.
            # indices_of_non_target_class = get_mixture_columns_for_class(1-target_class, priors_denominator)
            all_indices = get_mixture_columns_for_class([1] * len(target_class), priors_denominator)
            indices_of_non_target_class = [idx for idx in all_indices if idx not in indices_of_target_class]
            denominator = np.sum(prob[:, indices_of_non_target_class], axis=1)
            lrs[:, i] = numerator/denominator

        else: # sigmoid
            # TODO: Incorporate priors
            if len(target_classes) > 1:
                prob_target_class = prob[:, i].flatten()
                prob_target_class = np.reshape(prob_target_class, -1, 1)
                lrs[:, i] = prob_target_class / (1 - prob_target_class)
            else:  # when one target class it predicts either if it's the label
                # or if it's not the label.
                try:
                    lrs[:, i] = prob[:, 1] / prob[:, 0]
                except:
                    lrs[:, i] = np.reshape((prob / (1 - prob)), -1)

    lrs = np.where(lrs > 10 ** MAX_LR, 10 ** MAX_LR, lrs)
    lrs = np.where(lrs < 10 ** -MAX_LR, 10 ** -MAX_LR, lrs)

    return lrs


def get_mixture_columns_for_class(target_class, priors):
    """
    for the target_class, a vector of length n_single_cell_types with 1 or more 1's, give
    back the columns in the mixtures that contain one or more of these single cell types

    :param target_class: vector of length n_single_cell_types with at least one 1
    :param priors: vector of length n_single_cell_types with 0 or 1 to indicate single cell type has 0 or 1 prior,
    uniform assumed otherwise
    :return: list of ints, in [0, 2 ** n_cell_types]
    """

    def int_to_binary(i):
        binary = bin(i)[2:]
        while len(binary) < len(single_cell_types):
            binary = '0' + binary
        return np.flip([int(j) for j in binary]).tolist()

    def binary_admissable(binary, target_class, priors):
        """
        gives back whether the binary (string of 0 and 1 of length n_single_cell_types) has at least one of
        target_class in it, and all priors satisfied
        """
        if priors:
            for i in range(len(target_class)):
                # if prior is zero, the class should not occur
                if binary[i] == 1 and priors[i] == 0:
                    return False
                # if prior is one, the class should occur
                # as binary is zero it does not occur and return False
                if binary[i] == 0 and priors[i] == 1:
                    return False
        # at least one of the target class should occur
        if np.inner(binary, target_class)==0:
            return False
        return True

    return [i for i in range(2 ** len(single_cell_types)) if binary_admissable(int_to_binary(i), target_class, priors)]


def generate_lrs(model, mle, softmax, X_train, y_train, X_calib, y_calib, X_test, X_test_as_mixtures, X_mixtures, target_classes):
    """
    When softmax the model must be fitted on labels, whereas with sigmoid the model must be fitted on
    an nhot encoded vector representing the labels. Ensure that labels take the correct form, fit the
    model and predict the lrs before and after calibration for both X_test and X_mixtures.
    """

    if softmax: # y_train must be list with labels
        try:
            y_train = mle.nhot_to_labels(y_train)
        except: # already are labels
            pass
        # for DL model y_train must always be nhot encoded
        # TODO: Find better solution
        if isinstance(model._classifier, keras.engine.training.Model):
            y_train = np.eye(2 ** 8)[y_train]
    else: # y_train must be nhot encoded labels
        try:
            y_train = mle.labels_to_nhot(y_train)
        except: # already is nhot encoded
            pass
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(y_train[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    try: # y_calib must always be nhot encoded
        y_calib = mle.labels_to_nhot(y_calib)
    except: # already is nhot encoded
        pass

    model.fit_classifier(X_train, y_train)
    model.fit_calibration(X_calib, y_calib, target_classes)

    lrs_before_calib = model.predict_lrs(X_test, target_classes, with_calibration=False)
    lrs_after_calib = model.predict_lrs(X_test, target_classes)

    lrs_reduced_before_calib = model.predict_lrs(X_test_as_mixtures, target_classes, with_calibration=False)
    lrs_reduced_after_calib = model.predict_lrs(X_test_as_mixtures, target_classes)

    lrs_before_calib_mixt = model.predict_lrs(X_mixtures, target_classes, with_calibration=False)
    lrs_after_calib_mixt = model.predict_lrs(X_mixtures, target_classes)

    return lrs_before_calib, lrs_after_calib, lrs_reduced_before_calib, lrs_reduced_after_calib, \
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


def calculate_accuracy(model, mle, y_true, X, target_classes):
    """
    Predicts labels and ensures that both the true and predicted labels are nhot encoded.
    Calculates the accuracy.

    :return: accuracy: the set of labels predicted for a sample must *exactly* match the
        corresponding set of labels in y_true.
    """

    y_pred = model._classifier.predict(X)
    if isinstance(model._classifier, keras.engine.training.Model):
        # when the model predicts probabilities rather than classes
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

    if y_true.shape[1] != target_classes.shape[0] and y_pred.shape[1] == target_classes.shape[0]:
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_true = np.array([np.max(np.array(y_true[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    # TODO: return accuracy per target class
    return accuracy_score(y_true, y_pred)





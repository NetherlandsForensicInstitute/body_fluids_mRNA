import numpy as np

from sklearn.neural_network import MLPClassifier

from lir import KDECalibrator
from rna.analytics import get_mixture_columns_for_class


class MarginalClassifier():

    def __init__(self, random_state=0, classifier=MLPClassifier,
                 calibrator=KDECalibrator, MAX_LR=10, max_iter=200,
                 epsilon=1e-08):
        self._classifier = classifier(random_state=random_state, max_iter=max_iter, epsilon=epsilon)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for each target class
        """
        lrs_per_target_class = self.predict_lrs(X, target_classes)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator()
            loglrs = np.log10(lrs_per_target_class[:, i])
            labels = np.max(np.multiply(y_nhot, target_class), axis=1)
            self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)

    def predict_lrs(self, X, target_classes, with_calibration=False, priors_numerator=None, priors_denominator=None):
        """
        gives back an N x n_target_class array of LRs
        :param X: the N x n_features data
        :param target_classes:
        :param with_calibration:
        :param priors_numerator: vector of length n_single_cell_types, specifying 0 indicates we know this single cell type
        does not occur, specify 1 indicates we know this cell type certainly occurs, anything else assume implicit uniform
        distribution
        :param priors_denominator: vector of length n_single_cell_types, specifying 0 indicates we know this single cell type
        does not occur, specify 1 indicates we know this cell type certainly occurs, anything else assume implicit uniform
        distribution
        :return:
        """
        assert priors_numerator is None or type(priors_numerator) == list or type(priors_numerator) == np.ndarray
        assert priors_denominator is None or type(priors_denominator) == list or type(priors_denominator) == np.ndarray

        ypred_proba = self._classifier.predict_proba(X)
        lrs_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                              priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

        return lrs_per_target_class


def convert_prob_per_mixture_to_marginal_per_class(prob, target_classes, MAX_LR, priors_numerator=None, priors_denominator=None):
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
        assert sum(target_class) > 0, 'Nonexisting class in target_classes'

        # numerator
        indices_of_target_class = get_mixture_columns_for_class(target_class, priors_numerator)
        numerator = np.sum(prob[:, indices_of_target_class], axis=1)

        # denominator
        # TODO: Does this work when priors are defined?
        # TODO: Rewrite with priors.
        all_indices = get_mixture_columns_for_class([1] * len(target_class), priors_denominator)
        indices_of_non_target_class = [idx for idx in all_indices if idx not in indices_of_target_class]
        # indices_of_non_target_class = get_mixture_columns_for_class(1-target_class, priors_denominator)
        denominator = np.sum(prob[:, indices_of_non_target_class], axis=1)
        lrs[:, i] = numerator/denominator

    # TODO: Does this work when signal vals in stead of binary?
    lrs = np.where(lrs > 10 ** MAX_LR, 10 ** MAX_LR, lrs)
    lrs = np.where(lrs < 10 ** -MAX_LR, 10 ** -MAX_LR, lrs)

    return lrs

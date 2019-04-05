import numpy as np

from lir import KDECalibrator, Xn_to_Xy, Xy_to_Xn
from sklearn.neural_network import MLPClassifier

from rna.analytics import get_mixture_columns_for_class


class MarginalClassifier():

    def __init__(self, random_state=0, classifier=MLPClassifier,
                 calibrator=KDECalibrator, MAX_LR=10):
        self._classifier = classifier(random_state=random_state)
        self._calibrator = calibrator()
        self.MAX_LR = MAX_LR

    def fit(self, X, y):
        self._classifier.fit(X, y)
        return self

    def predict_lrs(self, X, target_classes, without_calibration=False, priors=None):
        """
        gives back an N x n_target_class array of LRs
        :param X: the N x n_features data
        :param target_classes:
        :param without_calibration:
        :param priors:
        :return:
        """
        if not without_calibration:
            ypred_proba = self._classifier.predict_proba(X)

            # mixtures = np.flip(np.unique(y_nhot, axis=0), axis=1) # select unique combinations of celltypes
            prob_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR)

            return prob_per_target_class
        else:
            pass

    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for all claqsses of interest.

        :param X:
        :param y:
        :param target_classes:
        :param index2string:
        :return:
        """
        self.fit(X, y_nhot)


        # TODO: outputs calibrated models

    def transform_calibration(self, ):
        pass


# TODO: Make everything h1_h2
def calibration_fit(h0_h1_probs, classes_map, Calibrator=KDECalibrator):
    """
    Get a calibrated model for each class based on one vs. all.

    :param h0_h1_probs:
    :param classes_map:
    :param Calibrator:
    :return:
    """
    calibrators_per_class = {}
    for j, celltype in enumerate(sorted(classes_map)):
        h0_h1_probs_celltype = h0_h1_probs[celltype]

        X, y = Xn_to_Xy(h0_h1_probs_celltype[0],
                        h0_h1_probs_celltype[1])

        calibrator = Calibrator()
        calibrators_per_class[celltype] = calibrator.fit(X, y)

    return calibrators_per_class


# TODO: In class
# TODO: Make everything h1_h2
def calibration_transform(h0_h1_probs_test, calibrators_per_class, classes_map):
    """
    Transforms the scores with the calibrated model for the correct class.

    :param h0_h1_probs_test:
    :param calibrators_per_class:
    :param classes_map:
    :return:
    """
    h0_h1_after_calibration = {}
    for celltype in sorted(classes_map):
        h0_h1_probs_celltype_test = h0_h1_probs_test[celltype]
        calibrator = calibrators_per_class[celltype]
        Xtest, ytest = Xn_to_Xy(h0_h1_probs_celltype_test[0],
                                h0_h1_probs_celltype_test[1])

        lr0, lr1 = Xy_to_Xn(calibrator.transform(Xtest), ytest)
        probs0 = lr0 / (1 + lr0)
        probs1 = lr1 / (1 + lr1)

        h0_h1_after_calibration[celltype] = (probs0, probs1)

    return h0_h1_after_calibration


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

    lrs = np.zeros((prob.shape[0], target_classes.shape[0]))
    for i, target_class in enumerate(target_classes):
        assert sum(target_class) > 0

        # numerator
        indices_of_target_class = get_mixture_columns_for_class(target_class, priors_numerator)
        numerator = np.sum(prob[:, indices_of_target_class][:, :, 0], axis=1)

        # denominator
        indices_of_target_class = get_mixture_columns_for_class(1-target_class, priors_denominator)
        denominator = np.sum(prob[:, indices_of_target_class][:, :, 0], axis=1)
        lrs[:, i] = numerator/denominator

    lrs = np.where(lrs > MAX_LR, MAX_LR, lrs)
    lrs = np.where(lrs < -MAX_LR, -MAX_LR, lrs)
    return lrs

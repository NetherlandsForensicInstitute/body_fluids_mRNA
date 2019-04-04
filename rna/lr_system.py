import numpy as np

from sklearn.neural_network import MLPClassifier

from lir import KDECalibrator
from rna.analytics import convert_prob_per_mixture_to_marginal_per_class

class MarginalClassifier():

    def __init__(self, random_state=0, classifier=MLPClassifier,
                 calibrator=KDECalibrator, MAX_LR=10):
        self._classifier = classifier(random_state=random_state)
        self._calibrator = calibrator()
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR


    def fit(self, X_train, y_train):
        """
        Trains classifier
        """
        self._classifier.fit(X_train, y_train)
        return self


    def fit_calibration(self, X_calib, y_nhot_calib, target_classes):
        """
        Makes calibrated model for each target class
        """
        prob_per_target_class = self.predict_proba(X_calib, y_nhot_calib, target_classes)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator
            probs = prob_per_target_class[:, i]
            labels = np.max(np.multiply(y_nhot_calib, target_class.T), axis=1)
            # TODO: Other type of key?
            self._calibrators_per_target_class[str(target_class)] = calibrator.fit(probs, labels)

        return self


    def predict_proba(self, X, y_nhot, target_classes, for_calibration=False):
        """
        Predicts probability for all n_mixtures and converts them to marginal probabilities
        for each target class. If for_calibration returns the calibrated probabilities.
        """
        ypred_proba = self._classifier.predict_proba(X)

        mixtures = np.flip(np.unique(y_nhot, axis=0), axis=1) # select unique combinations of celltypes
        prob_per_target_class = \
            convert_prob_per_mixture_to_marginal_per_class(ypred_proba, mixtures, target_classes, self.MAX_LR)

        if for_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                lrs_per_target_class = calibrator.transform(prob_per_target_class[:, i])
                prob_per_target_class[:, i] = lrs_per_target_class / (1 + lrs_per_target_class)

        return prob_per_target_class
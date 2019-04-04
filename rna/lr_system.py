import numpy as np

from lir import KDECalibrator, Xn_to_Xy, Xy_to_Xn
from sklearn.neural_network import MLPClassifier

from rna.analytics import convert_prob_per_mixture_to_marginal_per_class

class MarginalClassifier():

    def __init__(self, random_state=0, classifier=MLPClassifier,
                 calibrator=KDECalibrator, MAX_LR=10):
        self._classifier = classifier(random_state=random_state)
        self._calibrator = calibrator()
        self.MAX_LR = MAX_LR

    def fit(self, X, y):
        self._classifier.fit(X, y)
        return self


    def predict_proba(self, X, y_nhot, target_classes, for_calibration=False):
        if not for_calibration:
            ypred_proba = self._classifier.predict_proba(X)

            mixtures = np.flip(np.unique(y_nhot, axis=0), axis=1) # select unique combinations of celltypes
            prob_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, mixtures, target_classes, self.MAX_LR)

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
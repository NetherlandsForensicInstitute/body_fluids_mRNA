from lir import KDECalibrator, Xn_to_Xy, Xy_to_Xn
from reimplementation import *

class MLPClassifierMarginal():

    def __init__(self, random_state=0):
        self._mlp = MLPClassifier(random_state=random_state)

    def fit(self, X, y):
        """
        Train MLP model.
        """
        self._mlp.fit(X, y)
        return self


    def predict(self, X):
        """
        """
        ypred = self._mlp.predict(X)
        return ypred


    def predict_proba_per_class(self, Xtest, y_n_hot, labels_in_class, classes_map, classes_map_full, MAX_LR):
        """
        Predicts probabilties per class and returns probabilities for one class and the other in separate lists.
        """
        ypred_proba = self._mlp.predict_proba(Xtest)

        h0_h1_probs_per_class = {}
        # marginal for each single class sample
        prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
            ypred_proba, labels_in_class, classes_map, MAX_LR)

        for idx, (celltype, i_celltype) in enumerate(sorted(classes_map.items())):
            i_celltype_full = classes_map_full[celltype]
            total_proba = prob_per_class[:, i_celltype]
            if sum(total_proba) > 0:
                probas_without_cell_type = total_proba[y_n_hot[:, i_celltype_full] == 0]
                probas_with_cell_type = total_proba[y_n_hot[:, i_celltype_full] == 1]
                h0_h1_probs_per_class[celltype] = (probas_with_cell_type, probas_without_cell_type)

        return h0_h1_probs_per_class


    def predict_proba(self, Xtest, labels_in_class, classes_map, MAX_LR):
        """
        Predicts probabilties per class.
        """
        ypred_proba = self._mlp.predict_proba(Xtest)

        prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
            ypred_proba, labels_in_class, classes_map, MAX_LR)

        return prob_per_class


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
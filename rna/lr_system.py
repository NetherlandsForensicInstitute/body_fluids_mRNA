from functools import partial

import numpy as np
# import tensorflow as tf
# from keras import Input, Model
# from keras.layers import Dense, Dropout
from lir import LogitCalibrator, ELUBbounder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from xgboost import XGBClassifier

from rna.constants import single_cell_types


class MarginalClassifier():
    def fit_calibration(self, X, y_nhot, target_classes, calibration_on_loglrs=True):
        """
        Makes calibrated model for each target class
        :param calibration_on_loglrs:
        """
        lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator()
            # np.max ensures that the nhot matrix is converted into a nhot list with the
            # lrs from the relevant target classes coded as a 1.
            labels = np.max(np.multiply(y_nhot, target_class), axis=1)
            if calibration_on_loglrs:
                loglrs = np.log10(lrs_per_target_class[:, i]).reshape(-1, 1)
                # loglrs = np.nan_to_num(np.log10(lrs_per_target_class[:, i]).reshape(-1, 1), nan=-self.MAX_LR-1, posinf=self.MAX_LR, neginf=-self.MAX_LR)
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
            else:
                probs = np.nan_to_num(lrs_per_target_class[:, i] / (1 + lrs_per_target_class[:, i]))
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(probs.reshape(-1, 1), labels)


    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True,
                    calibration_on_loglrs=True):
        """
        gives back an N x n_target_class array of LRs
        :param calibration_on_loglrs:
        :param X: the N x n_features data
        :param target_classes: vector of length n_single_cell_types with at least one 1
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

        lrs_per_target_class = None
        try:
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                      priors_numerator, priors_denominator)
        except AttributeError:
            ypred_proba = self._classifier.predict(X)

            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                      priors_numerator, priors_denominator)


        if with_calibration:
            try:
                for i, target_class in enumerate(target_classes):
                    calibrator = self._calibrators_per_target_class[str(target_class)]
                    if calibration_on_loglrs:
                        loglrs_for_target_class = np.nan_to_num(np.log10(lrs_per_target_class[:, i]), nan=-self.MAX_LR-1, posinf=self.MAX_LR, neginf=-self.MAX_LR)
                        lrs_per_target_class[:, i] = calibrator.transform(loglrs_for_target_class.reshape(-1, 1))
                    else:
                        probs_for_target_class = np.nan_to_num(lrs_per_target_class[:, i] / (1 + lrs_per_target_class[:, i]), nan=-self.MAX_LR-1, posinf=self.MAX_LR, neginf=-self.MAX_LR)
                        lrs_per_target_class[:, i] = calibrator.transform(probs_for_target_class.reshape(-1, 1))
            except AttributeError:
                lrs_per_target_class = lrs_per_target_class

        return np.nan_to_num(lrs_per_target_class, nan=10**(-self.MAX_LR-1), posinf=10**self.MAX_LR, neginf=10**(-self.MAX_LR))



class MarginalMLPClassifier(MarginalClassifier):
    def __init__(self, calibrator=LogitCalibrator, activation='relu',
                 random_state=0, max_iter=500, MAX_LR=10):
        self._classifier = MLPClassifier(activation=activation, random_state=random_state, max_iter=max_iter)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        if self._classifier.activation == 'logistic':
            if y.shape[1] == 1:
                y = np.ravel(y)
        self._classifier.fit(X, y)


class MarginalRFClassifier(MarginalClassifier):
    def __init__(self, calibrator=LogitCalibrator, multi_label='ovr', MAX_LR=10):
        if multi_label=='ovr':
            self._classifier = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', max_depth=3))
        else:
            self._classifier = RandomForestClassifier(class_weight='balanced', max_depth=3)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        # if self._classifier.activation == 'logistic':
        #     if y.shape[1] == 1:
        #         y = np.ravel(y)
        self._classifier.fit(X, y)


class MarginalSVMClassifier(MarginalClassifier):

    def __init__(self, calibrator=LogitCalibrator, multi_label='ovr', MAX_LR=10):
        if multi_label=='ovr':
            self._classifier = OneVsRestClassifier(SVC(probability=True,
                class_weight='balanced'))
        else:
            self._classifier = SVC(class_weight='balanced', probability=True)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)


class MarginalMLRClassifier(MarginalClassifier):

    def __init__(self, random_state=0, calibrator=LogitCalibrator,
                 multi_class='ovr', solver='liblinear', MAX_LR=10):
        if multi_class == 'ovr':
            self._classifier = OneVsRestClassifier(LogisticRegression(multi_class=multi_class, solver=solver, class_weight='balanced'))
        else:
            self._classifier = LogisticRegression(random_state=random_state, solver=solver, multi_class=multi_class, class_weight='balanced')
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def get_coefficients(self, t, target_class):
        """
        Returns the intercept and coefficients, adjust for the calibrator.
        Only works if calibrator is none or logistic regression as well and we use one vs rest.
        :param t:
        :param target_class:
        :return:
        """
        if len(self._classifier.coef_) == 2 ** 8:
            # TODO: Is this correct for MLP with softmax?
            # NO
            # the marginal takes the sum over many probabilities. taking the log does not yield anything nice it seems
            # (although the mean will probably correlate)
            return None, None
            # indices_target_class = get_mixture_columns_for_class(target_class, None)
            # intercept = np.mean(model._classifier.intercept_[indices_target_class])
            # coefficients = np.mean(model._classifier.coef_[indices_target_class, :], axis=0)
        else:
            intercept = self._classifier.intercept_[t, :].squeeze() / np.log(10)
            coefficients = self._classifier.coef_[t, :].squeeze() / np.log(10)

        if self._calibrator:
            beta1 = self._calibrators_per_target_class[str(target_class)]._logit.coef_[0][0] / np.log(10)
            beta0 = self._calibrators_per_target_class[str(target_class)]._logit.intercept_[0] / np.log(10)
            intercept = intercept * beta1 + beta0
            coefficients = coefficients * beta1
        return intercept, coefficients


class MarginalXGBClassifier(MarginalClassifier):

    def __init__(self, method='softmax', calibrator=LogitCalibrator,
                 MAX_LR=10):
        if method == 'softmax':
            self._classifier = XGBClassifier(class_weight='balanced')
        elif method == 'sigmoid':
            self._classifier = OneVsRestClassifier(XGBClassifier(class_weight='balanced'))
        self.method = method
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

        if self.method == 'softmax':
            self.n_trees = len(self._classifier.get_booster().get_dump())
            # import matplotlib.pyplot as plt
            # from xgboost import plot_tree
            # import graphviz

            # plot_tree(self._classifier, num_trees=0)
            # plt.show()

        elif self.method == 'sigmoid':
            self.n_trees = len(self._classifier._first_estimator.get_booster().get_dump())
            # import matplotlib.pyplot as plt
            # from xgboost import plot_tree
            # import graphviz
            #
            # plot_tree(self._classifier._first_estimator, num_trees=10)
            # plt.show()


class MarginalDLClassifier(MarginalClassifier):

    def __init__(self, n_classes, activation_layer, optimizer, loss, epochs, units=80, n_features=15,
                 calibrator=partial(ELUBbounder, first_step_calibrator=LogitCalibrator()), MAX_LR=10):
        self.units = units
        self.n_classes = n_classes
        self.n_features = n_features
        self.activation_layer = activation_layer
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self._classifier = self.create_model()
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def build_model(self, units, n_classes, n_features, activation_layer):
        """
        Builds deep learning model

        :param units: (relative) number of units
        :param n_classes number of classes
        :param n_features: number of features
        :return: a keras model
        """
        drop = 0.05

        # inout shape
        x = Input(shape=(n_features,))
        # flatten input shape (i.e. remove the ,1)
        # first dense (hidden) layer
        cnn = Dense(units // 4, activation="sigmoid")(x)
        # dropout
        cnn = Dropout(rate=drop)(cnn)
        # second dense (hidden) layer
        cnn = Dense(units, activation="sigmoid")(cnn)

        # # adjusted DL:
        # input=500
        # drop = 0.5
        # x = Input(shape=(n_features, ))
        # cnn = Dense(units//4, activation="sigmoid")(x)
        # cnn = Dropout(rate=drop)(cnn)
        # cnn = Dense(units, activation="sigmoid")(cnn)
        # cnn = Dense(units//2, activation="sigmoid")(cnn)

        # output layer (corresponding to the number of classes)
        y = Dense(n_classes, activation=activation_layer)(cnn)

        # define inputs and outputs of the model
        model = Model(inputs=x, outputs=y)

        return model

    def compile_model(self, model, optimizer, loss):
        """
        compile a keras model using an optimizer and a loss function

        :param model: a keras model
        :param optimizer: a string or optimizer class that is supported by keras
        :param loss: a string or loss class that is supported by keras
        """
        model.compile(optimizer=optimizer, loss=loss, metrics=[self._accuracy_em])

    def create_model(self):
        """
        Create keras/tf model based on the number of classes, features and the the number of units in the model

        :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
        :param config: confidence object with specific information regarding the data
        :param n_classes: number of classes in the output layer
        :return: A compiled keras model
        """
        # build model
        model = self.build_model(units=self.units, n_classes=self.n_classes, n_features=self.n_features,
                                 activation_layer=self.activation_layer)
        # compile model
        self.compile_model(model, optimizer=self.optimizer, loss=self.loss)
        # model.summary()
        return model

    def _accuracy_exact_match(self, y_true, y_pred, threshold: float = .5):
        """
        Custom keras metric that mirrors the sklearn.metrics.accuracy_score, that is only samples that have the correct
        labels for each class are scored as 1. If not the sample is scored as 0.
        From: https://stackoverflow.com/questions/46799261/how-to-create-an-exact-match-eval-metric-op-for-tensorflow
        :param y_true: Tensor with the the true labels
        :param y_pred: Tensor with the predicted labels
        :param threshold: Threshold  used to classify a prediction as 1/0
        :return: float that represents the accuracy
        """
        # check if prediction are above threshold
        predictions = tf.to_float(tf.greater_equal(y_pred, threshold))
        # check if predictions match ground truth
        pred_match = tf.equal(predictions, tf.round(y_true))
        # reduce to mean
        exact_match = tf.reduce_min(tf.to_float(pred_match), axis=1)

        return exact_match

    def _accuracy_em(self, *args):
        """
        wrapper for _accuracy_exact_match
        :param args: input from metric evaluation provided by keras
        :return: float that represents the accuracy
        """
        return tf.reduce_mean(self._accuracy_exact_match(*args))


    def fit_classifier(self, X, y):
        self._classifier.fit(X, y, epochs=self.epochs, verbose=0)







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

        if prob.shape[1] == 2 ** target_classes.shape[1]:  # lps
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
            lrs[:, i] = numerator / denominator

        else:  # sigmoid
            if len(target_classes) > 1:
                prob_target_class = prob[:, i].flatten()
                # prob_target_class = np.reshape(prob_target_class, (-1, 1))
                lrs[:, i] = prob_target_class / (1 - prob_target_class)
            else:
                # When only one target class some classifiers predict the positive and negative label (i.e. output two probs)
                # and others predict only the probability of the positive label. With the following try and except statement
                # catch this.
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
        if np.inner(binary, target_class) == 0:
            return False
        return True

    return [i for i in range(2 ** len(single_cell_types)) if binary_admissable(int_to_binary(i), target_class, priors)]

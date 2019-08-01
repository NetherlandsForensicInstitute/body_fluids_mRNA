import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from keras import Input, Model
from keras.layers import Dense, Dropout

from xgboost import XGBClassifier

from lir import KDECalibrator
from rna.constants import single_cell_types


class MarginalMLPClassifier():

    def __init__(self, calibrator=KDECalibrator, activation='relu', random_state=0, max_iter=500, MAX_LR=10):
        self._classifier = MLPClassifier(activation=activation, random_state=random_state, max_iter=max_iter)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        if self._classifier.activation == 'logistic':
            if y.shape[1] == 1:
                y = np.ravel(y)
        self._classifier.fit(X, y)

    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for each target class
        """
        try:
            lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrator()
                loglrs = np.log10(lrs_per_target_class[:, i])
                # np.max ensures that the nhot matrix is converted into a nhot list with the
                # relevant target classes coded as a 1.
                labels = np.max(np.multiply(y_nhot, target_class), axis=1)
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
        except TypeError or ValueError:
            for target_class in target_classes:
                self._calibrators_per_target_class[str(target_class)] = None

    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
        """
        gives back an N x n_target_class array of LRs
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

        try:
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR, priors_numerator, priors_denominator)

        except ValueError:
            lrs_per_target_class = None

        if with_calibration:
            try:
                for i, target_class in enumerate(target_classes):
                    calibrator = self._calibrators_per_target_class[str(target_class)]
                    loglrs_per_target_class = np.log10(lrs_per_target_class)
                    lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])
            except AttributeError:
                lrs_per_target_class = lrs_per_target_class

        return lrs_per_target_class


class MarginalMLRClassifier():

    def __init__(self, random_state=0, calibrator=KDECalibrator, multi_class='ovr', solver='liblinear', MAX_LR=10):
        if multi_class == 'ovr':
            self._classifier = OneVsRestClassifier(LogisticRegression(multi_class=multi_class, solver=solver))
        else:
            self._classifier = LogisticRegression(random_state=random_state, solver=solver, multi_class=multi_class)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for each target class
        """
        try:
            lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrator()
                loglrs = np.log10(lrs_per_target_class[:, i])
                labels = np.max(np.multiply(y_nhot, target_class), axis=1)
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
        except TypeError or ValueError:
            for target_class in target_classes:
                self._calibrators_per_target_class[str(target_class)] = None

    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
        """
        gives back an N x n_target_class array of LRs
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

        try:
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR, priors_numerator, priors_denominator)

        except ValueError:
            lrs_per_target_class = None

        if with_calibration:
            try:
                for i, target_class in enumerate(target_classes):
                    calibrator = self._calibrators_per_target_class[str(target_class)]
                    loglrs_per_target_class = np.log10(lrs_per_target_class)
                    lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])
            except AttributeError:
                lrs_per_target_class = lrs_per_target_class

        return lrs_per_target_class


class MarginalXGBClassifier():

    def __init__(self, method='softmax', calibrator=KDECalibrator, MAX_LR=10):
        if method == 'softmax':
            self._classifier = XGBClassifier()
        elif method == 'sigmoid':
            self._classifier = OneVsRestClassifier(XGBClassifier())
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.MAX_LR = MAX_LR

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for each target class
        """
        try:
            lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrator()
                loglrs = np.log10(lrs_per_target_class[:, i])
                labels = np.max(np.multiply(y_nhot, target_class), axis=1)
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
        except TypeError or ValueError:
            for target_class in target_classes:
                self._calibrators_per_target_class[str(target_class)] = None

    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
        """
        gives back an N x n_target_class array of LRs
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

        try:
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR, priors_numerator, priors_denominator)

        except ValueError:
            lrs_per_target_class = None

        if with_calibration:
            try:
                for i, target_class in enumerate(target_classes):
                    calibrator = self._calibrators_per_target_class[str(target_class)]
                    loglrs_per_target_class = np.log10(lrs_per_target_class)
                    lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])
            except AttributeError:
                lrs_per_target_class = lrs_per_target_class

        return lrs_per_target_class


class MarginalDLClassifier():

    def __init__(self, n_classes, activation_layer, optimizer, loss, epochs, units=80, n_features=15,
                 calibrator=KDECalibrator, MAX_LR=10):
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
        # set drop out
        drop = 0.05

        # inout shape
        x = Input(shape=(n_features, ))
        # flatten input shape (i.e. remove the ,1)
        # first dense (hidden) layer
        cnn = Dense(units//4, activation="sigmoid")(x)
        # dropout
        cnn = Dropout(rate=drop)(cnn)
        # second dense (hidden) layer
        cnn = Dense(units, activation="sigmoid")(cnn)

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
        model.compile(optimizer=optimizer, loss=loss)


    def create_model(self):
        """
        Create keras/tf model based on the number of classes, features and the the number of units in the model

        :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
        :param config: confidence object with specific information regarding the data
        :param n_classes: number of classes in the output layer
        :return: A compiled keras model
        """
        # build model
        model = self.build_model(units=self.units, n_classes=self.n_classes, n_features=self.n_features, activation_layer=self.activation_layer)
        # compile model
        self.compile_model(model, optimizer=self.optimizer, loss=self.loss)
        model.summary()
        return model


    def fit_classifier(self, X, y):
        self._classifier.fit(X, y, epochs=self.epochs)


    def fit_calibration(self, X, y_nhot, target_classes):
        """
        Makes calibrated model for each target class
        """
        try:
            lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrator()
                loglrs = np.log10(lrs_per_target_class[:, i])
                labels = np.max(np.multiply(y_nhot, target_class), axis=1)
                self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
        except TypeError or ValueError:
            for target_class in target_classes:
                self._calibrators_per_target_class[str(target_class)] = None

    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
        assert priors_numerator is None or type(priors_numerator) == list or type(priors_numerator) == np.ndarray
        assert priors_denominator is None or type(priors_denominator) == list or type(priors_denominator) == np.ndarray

        try:
            ypred_proba = self._classifier.predict(X)
            lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                      priors_numerator, priors_denominator)
        except ValueError:
            lrs_per_target_class = None

        if with_calibration:
            try:
                for i, target_class in enumerate(target_classes):
                    calibrator = self._calibrators_per_target_class[str(target_class)]
                    loglrs_per_target_class = np.log10(lrs_per_target_class)
                    lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])
            except AttributeError:
                lrs_per_target_class = lrs_per_target_class

        return lrs_per_target_class


# DL supposed to be the same as MLP with basic settings
# created this class to check for this
# class MarginalDLClassifier():
#
#     def __init__(self, n_classes, activastion_layer, optimizer, loss, epochs, units=100, n_features=15,
#                  calibrator=KDECalibrator, MAX_LR=10):
#         self.units = units
#         self.n_classes = n_classes
#         self.n_features = n_features
#         self.activation_layer = activation_layer
#         self.optimizer = optimizer
#         self.loss = loss
#         self.epochs = epochs
#         self._classifier = self.create_model()
#         self._calibrator = calibrator
#         self._calibrators_per_target_class = {}
#         self.MAX_LR = MAX_LR
#
#
#     def build_model(self, units, n_classes, n_features, activation_layer):
#         """
#         Builds deep learning model
#
#         :param units: (relative) number of units
#         :param n_classes number of classes
#         :param n_features: number of features
#         :return: a keras model
#         """
#         # inout shape
#         x = Input(shape=(n_features, ))
#         # flatten input shape (i.e. remove the ,1)
#         # first dense (hidden) layer
#         cnn = Dense(units, activation="relu")(x)
#
#         # output layer (corresponding to the number of classes)
#         y = Dense(n_classes, activation=activation_layer, kernel_regularizer=regularizers.l2(0.001))(cnn)
#
#         # define inputs and outputs of the model
#         model = Model(inputs=x, outputs=y)
#
#         return model
#
#
#     def compile_model(self, model, optimizer, loss):
#         """
#         compile a keras model using an optimizer and a loss function
#
#         :param model: a keras model
#         :param optimizer: a string or optimizer class that is supported by keras
#         :param loss: a string or loss class that is supported by keras
#         """
#         model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
#
#
#     def create_model(self):
#         """
#         Create keras/tf model based on the number of classes, features and the the number of units in the model
#
#         :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
#         :param config: confidence object with specific information regarding the data
#         :param n_classes: number of classes in the output layer
#         :return: A compiled keras model
#         """
#         # build model
#         model = self.build_model(units=self.units, n_classes=self.n_classes, n_features=self.n_features, activation_layer=self.activation_layer)
#         # compile model
#         self.compile_model(model, optimizer=self.optimizer, loss=self.loss)
#         model.summary()
#         return model
#
#
#     def fit_classifier(self, X, y):
#         self._classifier.fit(X, y, epochs=self.epochs, batch_size=200)
#
#
#     def fit_calibration(self, X, y_nhot, target_classes):
#         """
#         Makes calibrated model for each target class
#         """
#         lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)
#
#         for i, target_class in enumerate(target_classes):
#             calibrator = self._calibrator()
#             loglrs = np.log10(lrs_per_target_class[:, i])
#             labels = np.max(np.multiply(y_nhot, target_class), axis=1)
#             self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)
#
#
#     def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
#         assert priors_numerator is None or type(priors_numerator) == list or type(priors_numerator) == np.ndarray
#         assert priors_denominator is None or type(priors_denominator) == list or type(priors_denominator) == np.ndarray
#
#         ypred_proba = self._classifier.predict(X)
#         lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
#                                                                   priors_numerator, priors_denominator)
#
#         if with_calibration:
#             for i, target_class in enumerate(target_classes):
#                 calibrator = self._calibrators_per_target_class[str(target_class)]
#                 loglrs_per_target_class = np.log10(lrs_per_target_class)
#                 lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])
#
#         return lrs_per_target_class


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
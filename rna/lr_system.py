import os

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from keras import Input, Model
from keras import regularizers
from keras.layers import Dense, Dropout

from xgboost import XGBClassifier

from lir import KDECalibrator
from lir.plotting import makeplot_hist_density
from rna.analytics import convert_prob_to_marginal_per_class, generate_lrs
from rna.plotting import plot_histogram_log_lr
from rna.utils import bool2str_binarize, bool2str_softmax


class MarginalMLPClassifier():

    def __init__(self, calibrator=KDECalibrator, activation='relu', random_state=0, max_iter=200, MAX_LR=10):
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
        lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator()
            loglrs = np.log10(lrs_per_target_class[:, i])
            # np.max ensures that the nhot matrix is converted into a nhot list with the
            # relevant target classes coded as a 1.
            labels = np.max(np.multiply(y_nhot, target_class), axis=1)
            self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)

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

        ypred_proba = self._classifier.predict_proba(X)
        lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR, priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

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
        lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator()
            loglrs = np.log10(lrs_per_target_class[:, i])
            labels = np.max(np.multiply(y_nhot, target_class), axis=1)
            self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)

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

        ypred_proba = self._classifier.predict_proba(X)
        lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR, priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

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
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])


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
        lrs_per_target_class = self.predict_lrs(X, target_classes, with_calibration=False)

        for i, target_class in enumerate(target_classes):
            calibrator = self._calibrator()
            loglrs = np.log10(lrs_per_target_class[:, i])
            labels = np.max(np.multiply(y_nhot, target_class), axis=1)
            self._calibrators_per_target_class[str(target_class)] = calibrator.fit(loglrs, labels)


    def predict_lrs(self, X, target_classes, priors_numerator=None, priors_denominator=None, with_calibration=True):
        assert priors_numerator is None or type(priors_numerator) == list or type(priors_numerator) == np.ndarray
        assert priors_denominator is None or type(priors_denominator) == list or type(priors_denominator) == np.ndarray

        ypred_proba = self._classifier.predict(X)
        lrs_per_target_class = convert_prob_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                  priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

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


def perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_augmented, y_train_nhot_augmented,
                     X_calib_augmented, y_calib_nhot_augmented, X_test_augmented, y_test_nhot_augmented,
                     X_test_as_mixtures_augmented, X_mixtures, target_classes, save_hist=False):

    classifier = models[0]
    with_calibration = models[1]

    model = model_with_correct_settings(classifier, softmax, n_classes=target_classes.shape[0])

    if with_calibration: # with calibration
        model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(model, mle, softmax, X_train_augmented, y_train_nhot_augmented, X_calib_augmented,
                         y_calib_nhot_augmented, X_test_augmented, X_test_as_mixtures_augmented, X_mixtures,
                         target_classes, y_test_nhot_augmented)

        if save_hist:
            plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, density=True,
                                  savefig=os.path.join('scratch',
                                                       'hist_before_{}_{}_{}_{}'.format(n, bool2str_binarize(binarize),
                                                                                        bool2str_softmax(softmax),
                                                                                        classifier)))
            plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder, title='after',
                                  density=True, savefig=os.path.join('scratch', 'hist_after_{}_{}_{}_{}'.format(n,
                                                                                                                bool2str_binarize(
                                                                                                                    binarize),
                                                                                                                bool2str_softmax(
                                                                                                                    softmax),
                                                                                                                classifier)))
            makeplot_hist_density(model.predict_lrs(X_calib_augmented, target_classes, with_calibration=False),
                              y_calib_nhot_augmented, model._calibrators_per_target_class, target_classes,
                              label_encoder, savefig=os.path.join('scratch', 'kernel_density_estimation{}_{}_{}_{}'.format(n, bool2str_binarize(binarize), bool2str_softmax(softmax), classifier)))

    else: # no calibration
        model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(model, mle, softmax, np.concatenate((X_train_augmented, X_calib_augmented), axis=0),
                         np.concatenate((y_train_nhot_augmented, y_calib_nhot_augmented), axis=0), np.array([]),
                         np.array([]), X_test_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes,
                         y_test_nhot_augmented)

        if save_hist:
            plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, density=True,
                                  savefig=os.path.join('scratch',
                                                       'hist_before_{}_{}_{}_{}'.format(n, bool2str_binarize(binarize),
                                                                                        bool2str_softmax(softmax),
                                                                                        classifier)))

    return model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
           lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt
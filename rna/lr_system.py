import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from lir import KDECalibrator
from rna.analytics import get_mixture_columns_for_class

from keras import Input, Model
from keras.layers import Dense, Dropout


class MarginalClassifier():

    def __init__(self, random_state=0, classifier=MLPClassifier, calibrator=KDECalibrator, activation='relu', MAX_LR=10):
        self._classifier = classifier(activation=activation, random_state=random_state)
        self._calibrator = calibrator
        self._calibrators_per_target_class = {}
        self.activation = activation
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
        except ValueError:
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

        if self.activation == 'relu': # lps
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                                      priors_numerator, priors_denominator)
        elif self.activation == 'logistic': # sigmoids
            ypred_proba = self._classifier.predict_proba(X)
            lrs_per_target_class = convert_prob_per_single_to_target_classes(ypred_proba, target_classes, self.MAX_LR,
                                                                             priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

        return lrs_per_target_class


class MLRClassifier():

    def __init__(self, random_state=0, classifier=LogisticRegression, calibrator=KDECalibrator, solver='liblinear',
                 multi_class='ovr', MAX_LR=10):
        self._classifier = classifier(random_state=random_state, solver=solver, multi_class=multi_class)
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
            if ypred_proba.shape[1] > 8:
                lrs_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                                          priors_numerator, priors_denominator)
            else:
                lrs_per_target_class = convert_prob_per_single_to_target_classes(ypred_proba, target_classes, self.MAX_LR,
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


class myXGBClassifier():

    def __init__(self, classifier=XGBClassifier, calibrator=KDECalibrator, MAX_LR=10):
        self._classifier = classifier()
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
        except ValueError:
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

        ypred_proba = self._classifier.predict_proba(X)
        lrs_per_target_class = convert_prob_per_mixture_to_marginal_per_class(ypred_proba, target_classes, self.MAX_LR,
                                                                                  priors_numerator, priors_denominator)

        if with_calibration:
            for i, target_class in enumerate(target_classes):
                calibrator = self._calibrators_per_target_class[str(target_class)]
                loglrs_per_target_class = np.log10(lrs_per_target_class)
                lrs_per_target_class[:, i] = calibrator.transform(loglrs_per_target_class[:, i])

        return lrs_per_target_class


class DLClassifier():

    def __init__(self, units=80, n_classes=8, n_features=19):
        self.units = units
        self.n_classes = n_classes
        self.n_features = n_features
        self._classifier = self.create_model()

    def build_model(self, units: int, n_classes: int, n_features: int) -> Model:
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
        y = Dense(n_classes, activation="sigmoid")(cnn)

        # define inputs and outputs of the model
        model = Model(inputs=x, outputs=y)

        return model


    def compile_model(self, model: Model, optimizer: str = "adam", loss: str = "binary_crossentropy") -> None:
        """
        compile a keras model using an optimizer and a loss function

        :param model: a keras model
        :param optimizer: a string or optimizer class that is supported by keras
        :param loss: a string or loss class that is supported by keras
        """
        model.compile(optimizer=optimizer, loss=loss)


    def create_model(self) -> Model:
        """
        Create keras/tf model based on the number of classes, features and the the number of units in the model

        :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
        :param config: confidence object with specific information regarding the data
        :param n_classes: number of classes in the output layer
        :return: A compiled keras model
        """
        # build model
        model = self.build_model(units=self.units, n_classes=self.n_classes, n_features=self.n_features)
        # compile model
        self.compile_model(model)
        print(model.summary)

        return model


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
        assert sum(target_class) > 0, 'No cell type given as target class'

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

    lrs = np.where(lrs > 10 ** MAX_LR, 10 ** MAX_LR, lrs)
    lrs = np.where(lrs < 10 ** -MAX_LR, 10 ** -MAX_LR, lrs)

    return lrs


def convert_prob_per_single_to_target_classes(prob, target_classes, MAX_LR, priors_numerator=None, priors_denominator=None):

    # TODO: Incorporate priors
    lrs = np.zeros((len(prob), len(target_classes)))
    for i, target_class in enumerate(target_classes):
        assert sum(target_class) > 0, 'No cell type given as target class'
        prob_target_class = prob[:, np.argwhere(target_class == 1).flatten()]
        if prob_target_class.shape[1] > 1:
            # more than one cell type
            # TODO: Correct to calculate the mean of the probability?
            prob_target_class = np.mean(prob_target_class, axis=1)
        prob_target_class = np.reshape(prob_target_class, -1, 1)
        lrs[:, i] = prob_target_class / (1 - prob_target_class)

    lrs = np.where(lrs > 10 ** MAX_LR, 10 ** MAX_LR, lrs)
    lrs = np.where(lrs < 10 ** -MAX_LR, 10 ** -MAX_LR, lrs)

    return lrs

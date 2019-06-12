import collections
import logging

import numpy as np
import sklearn
import sklearn.mixture

from . import calibration
from .util import Xn_to_Xy, Xy_to_Xn

LOG = logging.getLogger(__name__)


class CalibratedScorer:
    def __init__(self, scorer, calibrator, fit_calibrator=False):
        self.scorer = scorer
        self.calibrator = calibrator
        self.fit_calibrator = fit_calibrator

    def fit(self, X, y):
        self.scorer.fit_classifier(X, y)
        if self.fit_calibrator:
            p = self.scorer.predict_proba(X)
            self.calibrator.fit_classifier(p[:, 1], y)

    def predict_lr(self, X):
        X = self.scorer.predict_proba(X)[:, 1]
        return self.calibrator.transform(X)


class CalibratedScorerCV:
    def __init__(self, scorer, calibrator, n_splits):
        self.scorer = scorer
        self.calibrator = calibrator
        self.n_splits = n_splits

    def fit(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        Xcal = np.empty([0, 1])
        ycal = np.empty([0])
        for train_index, cal_index in kf.split(X, y):
            self.scorer.fit_classifier(X[train_index], y[train_index])
            p = self.scorer.predict_proba(X[cal_index])
            Xcal = np.append(Xcal, p[:, 1])
            ycal = np.append(ycal, y[cal_index])
        self.calibrator.fit_classifier(Xcal, ycal)

        self.scorer.fit_classifier(X, y)

    def predict_lr(self, X):
        scores = self.scorer.predict_proba(X)[:, 1]  # probability of class 1
        return self.calibrator.transform(scores)


LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


def calibrate_lr(point, calibrator):
    """
    Calculates a calibrated likelihood ratio (LR).

    P(E|H1) / P(E|H0)

    Parameters
    ----------
    point : float
        The point of interest for which the LR is calculated. This point is
        sampled from either class 0 or class 1.
    calibrator : a calibration function
        A calibration function which is fitted for two distributions.

    Returns
    -------
    tuple<3> of float
        A likelihood ratio, with positive values in favor of class 1, and the
        probabilities P(E|H0), P(E|H1).
    """
    lr = calibrator.transform(np.array(point))[0]
    return LR(lr, calibrator.p0[0], calibrator.p1[0])


LrStats = collections.namedtuple('LrStats',
                                 ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'avg_p0_class0', 'avg_p1_class0',
                                  'avg_p0_class1', 'avg_p1_class1', 'cllr_class0', 'cllr_class1', 'cllr', 'lr_class0',
                                  'lr_class1', 'cllr_min', 'cllr_cal'])


def calculate_cllr(lr_class0, lr_class1):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Parameters
    ----------
    lr_class0 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 0.
    lr_class1 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 1.

    Returns
    -------
    LrStats
        Likelihood ratio statistics.
    """
    assert len(lr_class0) > 0
    assert len(lr_class1) > 0

    def avg(*args):
        return sum(args) / len(args)

    def _cllr(lr0, lr1):
        with np.errstate(divide='ignore'):
            cllr0 = np.mean(np.log2(1 + lr0))
            cllr1 = np.mean(np.log2(1 + 1 / lr1))
            return avg(cllr0, cllr1), cllr0, cllr1

    if type(lr_class0[0]) == LR:
        avg_p0_class0 = avg(*[lr.p0 for lr in lr_class0])
        avg_p1_class0 = avg(*[lr.p1 for lr in lr_class0])
        avg_p0_class1 = avg(*[lr.p0 for lr in lr_class1])
        avg_p1_class1 = avg(*[lr.p1 for lr in lr_class1])
        lr_class0 = np.array([lr.lr for lr in lr_class0])
        lr_class1 = np.array([lr.lr for lr in lr_class1])
    else:
        if type(lr_class0) == list:
            lr_class0 = np.array(lr_class0)
            lr_class1 = np.array(lr_class1)

        avg_p0_class0 = None
        avg_p1_class0 = None
        avg_p0_class1 = None
        avg_p1_class1 = None

    avg_llr_class0 = np.mean(np.log2(1 / lr_class0))
    avg_llr_class1 = np.mean(np.log2(lr_class1))
    avg_llr = avg(avg_llr_class0, avg_llr_class1)

    cllr, cllr_class0, cllr_class1 = _cllr(lr_class0, lr_class1)

    irc = calibration.IsotonicCalibrator()
    lr, y = Xn_to_Xy(lr_class0 / (lr_class0 + 1), lr_class1 / (lr_class1 + 1))  # translate LRs to range 0..1
    lrmin = irc.fit_transform(lr, y)
    cllrmin, cllrmin_class0, cllrmin_class1 = _cllr(*Xy_to_Xn(lrmin, y))

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1,
                   cllr_class0, cllr_class1, cllr, lr_class0, lr_class1, cllrmin, cllr - cllrmin)


def apply_scorer(scorer, X):
    return scorer.predict_proba(X)[:, 1]  # probability of class 1


def scorebased_lr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X_disputed):
    """
    Trains a classifier, calibrates the outcome and calculates a LR for a single
    sample.

    P(E|H1) / P(E|H0)

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    X0_train : numpy array
        Training set of samples from class 0
    X1_train : numpy array
        Training set of samples from class 1
    X0_calibrate : numpy array
        Calibration set of samples from class 0
    X1_calibrate : numpy array
        Calibration set of samlpes from class 1
    X_disputed : numpy array
        Test samples of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_disputed`
    """
    Xtrain, ytrain = Xn_to_Xy(X0_train, X1_train)
    scorer.fit_classifier(Xtrain, ytrain)

    Xcal, ycal = Xn_to_Xy(X0_calibrate, X1_calibrate)
    calibrator.fit_classifier(scorer.predict_proba(Xcal)[:, 1], ycal)
    scorer = CalibratedScorer(scorer, calibrator)

    return scorer.predict_lr(X_disputed)


def calibrated_cllr(calibrator, class0_calibrate, class1_calibrate, class0_test=None, class1_test=None):
    Xcal, ycal = Xn_to_Xy(class0_calibrate, class1_calibrate)
    calibrator.fit_classifier(Xcal, ycal)

    use_calibration_set_for_test = class0_test is None
    if use_calibration_set_for_test:
        class0_test = class0_calibrate
        class1_test = class1_calibrate

    lrs0 = calibrator.transform(class0_test)
    lrs1 = calibrator.transform(class1_test)

    lrs0 = [LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1)]
    lrs1 = [LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1)]

    return calculate_cllr(lrs0, lrs1)


def scorebased_cllr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X0_test=None, X1_test=None):
    """
    Trains a classifier on a training set, calibrates the outcome with a
    calibration set, and calculates a LR (likelihood ratio) for all samples in
    a test set. Calculates a Cllr (log likelihood ratio cost) value for the LR
    values. If no test set is provided, the calibration set is used for both
    calibration and testing.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    density_function : function
        A density function which is used to deterimine the density of
        a classifier outcome when sampled from either of both classes.
    X0_train : numpy array
        Training set for class 0
    X1_train : numpy array
        Training set for class 1
    X0_calibrate : numpy array
        Calibration set for class 0
    X1_calibrate : numpy array
        Calibration set for class 1
    X0_test : numpy array
        Test set for class 0
    X1_test : numpy array
        Test set for class 1

    Returns
    -------
    float
        A likelihood ratio for `X_test`
    """
    if X0_test is None:
        LOG.debug('scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}'.format(
            train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0], cal1=X1_calibrate.shape[0]))
    else:
        LOG.debug(
            'scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}; test size: {test0}/{test1}'.format(
                train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0],
                cal1=X1_calibrate.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    scorer.fit_classifier(*Xn_to_Xy(X0_train, X1_train))

    class0_test = apply_scorer(scorer, X0_test) if X0_test is not None else None
    class1_test = apply_scorer(scorer, X1_test) if X0_test is not None else None

    return calibrated_cllr(calibrator, apply_scorer(scorer, X0_calibrate), apply_scorer(scorer, X1_calibrate),
                           class0_test, class1_test)


def scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed):
    scorer = CalibratedScorerCV(scorer, calibrator, n_splits=n_splits)
    scorer.fit(*Xn_to_Xy(X0_train, X1_train))
    return scorer.predict_lr(X_disputed)


def scorebased_cllr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X0_test, X1_test):
    LOG.debug('scorebased_cllr_kfold: training_size: {train0}/{train1}; test size: {test0}/{test1}'.format(
        train0=X0_train.shape[0], train1=X1_train.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    X_disputed = np.concatenate([X0_test, X1_test])
    lrs = scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed)

    assert X0_test.shape[0] + X1_test.shape[0] == len(lrs)
    lrs0 = lrs[:X0_test.shape[0]]
    lrs1 = lrs[X0_test.shape[0]:]

    lrs0 = [LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1)]
    lrs1 = [LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1)]

    return calculate_cllr(lrs0, lrs1)
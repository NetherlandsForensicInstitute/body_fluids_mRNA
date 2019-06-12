import logging
import math
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from .util import Xy_to_Xn, Xn_to_Xy


LOG = logging.getLogger(__name__)


class NormalizedCalibrator(BaseEstimator, TransformerMixin):
    """
    Normalizer for any calibration function.

    Scales the probability density function of a calibrator so that the
    probability mass is 1.
    """

    def __init__(self, calibrator, add_one=False, sample_size=100, value_range=(0, 1)):
        self.calibrator = calibrator
        self.add_one = add_one
        self.value_range = value_range
        self.step_size = (value_range[1] - value_range[0]) / sample_size

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        self.X0n = X0.shape[0]
        self.X1n = X1.shape[0]
        self.calibrator.fit_classifier(X, y)
        self.calibrator.transform(np.arange(self.value_range[0], self.value_range[1], self.step_size))
        self.p0mass = np.sum(self.calibrator.p0) / 100
        self.p1mass = np.sum(self.calibrator.p1) / 100
        return self

    def transform(self, X):
        self.calibrator.transform(X)
        self.p0 = self.calibrator.p0 / self.p0mass
        self.p1 = self.calibrator.p1 / self.p1mass
        if self.add_one:
            self.p0 = self.X0n / (self.X0n + 1) * self.p0 + 1 / self.X0n
            self.p1 = self.X1n / (self.X1n + 1) * self.p1 + 1 / self.X1n
        return self.p1 / self.p0

    def __getattr__(self, name):
        return getattr(self.calibrator, name)


class FractionCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of the distance of a score value to the
    extremes of its value range.
    """

    def __init__(self, value_range=(0, 1)):
        self.value_range = value_range

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        self._abs_points0 = np.abs(self.value_range[0] - X0)
        self._abs_points1 = np.abs(self.value_range[1] - X1)
        return self

    def density(self, X, class_value, points):
        X = np.abs(self.value_range[class_value] - X)
        return (1 + np.array([points[points >= x].shape[0] for x in X])) / (1 + len(points))

    def transform(self, X):
        self.p0 = self.density(X, 0, self._abs_points0)
        self.p1 = self.density(X, 1, self._abs_points1)
        return self.p1 / self.p0


class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def bandwidth_silverman(self, X):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        std = np.std(X)
        if std == 0:
            # can happen eg if std(X) = 0
            warnings.warn('silverman bandwidth cannot be calculated if standard deviation is 0', RuntimeWarning)
            LOG.info('found a silverman bandwidth of 0 (using dummy value)')
            std = 1

        v = math.pow(std, 5) / len(X) * 4. / 3
        return math.pow(v, .2)

    def bandwidth_scott(self, X):
        """
        Not implemented.
        """
        raise

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0 = self.bandwidth or self.bandwidth_silverman(X0)
        bandwidth1 = self.bandwidth or self.bandwidth_silverman(X1)

        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        self._base_value0 = 1. / X0.shape[0]
        self._base_value1 = 1. / X1.shape[0]
        return self

    def transform(self, X):
        X = X.reshape(-1, 1)
        self.p0 = self._base_value0 + np.exp(self._kde0.score_samples(X))
        self.p1 = self._base_value1 + np.exp(self._kde1.score_samples(X))
        return self.p1 / self.p0


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """

    def fit(self, X, y):
        X = X.reshape(-1, 1)
        self._logit = LogisticRegression(class_weight='balanced')
        self._logit.fit(X, y)
        return self

    def transform(self, X):
        X = self._logit.predict_proba(X)[:, 1]  # probability of class 1
        self.p0 = (1 - X)
        self.p1 = X
        return self.p1 / self.p0


class GaussianCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.
    """

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = GaussianMixture().fit(X0)
        self._model1 = GaussianMixture().fit(X1)
        self._base_value0 = 1. / X0.shape[0]
        self._base_value1 = 1. / X1.shape[0]
        return self

    def transform(self, X):
        X = X.reshape(-1, 1)
        self.p0 = self._base_value0 + np.exp(self._model0.score_samples(X))
        self.p1 = self._base_value1 + np.exp(self._model1.score_samples(X))
        return self.p1 / self.p0


class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses isotonic regression for interpolation.
    """

    def __init__(self, add_one=False):
        self.add_one = add_one
        self._ir = IsotonicRegression()

    def fit(self, X, y, **fit_params):
        X0, X1 = Xy_to_Xn(X, y)

        # prevent extreme LRs
        if ('add_one' in fit_params and fit_params['add_one']) or self.add_one:
            X0 = np.append(X0, 1)
            X1 = np.append(X1, 0)

        X0n = X0.shape[0]
        X1n = X1.shape[0]
        X, y = Xn_to_Xy(X0, X1)

        weight = np.concatenate([[X1n] * X0n, [X0n] * X1n])
        self._ir.fit(X, y, sample_weight=weight)

        return self

    def transform(self, X):
        if isinstance(X, np.matrix):
            X = X.A1

        posterior = self._ir.transform(X)

        self.p0 = (1 - posterior)
        self.p1 = posterior
        with np.errstate(divide='ignore'):
            return self.p1 / self.p0


class DummyCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. No calibration is applied. Instead, the score value is
    interpreted as a posterior probability of the value being sampled from
    class 1.
    """

    def fit(self, X, y=None, **fit_params):
        X0, X1 = Xy_to_Xn(X, y)
        self._base_value0 = 1. / X0.shape[0]
        self._base_value1 = 1. / X1.shape[0]

        return self

    def transform(self, X):
        self.p0 = self._base_value0 + (1 - X)
        self.p1 = self._base_value0 + X
        return self.p1 / self.p0
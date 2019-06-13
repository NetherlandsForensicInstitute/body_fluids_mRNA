import collections
import logging
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import lr
from .util import Xn_to_Xy

from rna.utils import vec2string

LOG = logging.getLogger(__name__)


def process_vector(preprocessor, *X):
    X = [ np.empty(shape=(0,X[0].shape[1])) if x is None else x for x in X ]
    X_all = np.concatenate(X)

    X_all = preprocessor.fit_transform(X_all)
    cursor = 0
    X_out = []
    for x in X:
        X_out.append(X_all[cursor:cursor+x.shape[0],:] if x.shape[0] > 0 else None)
        cursor += x.shape[0]

    return X_out


class AbstractCllrEvaluator:
    def __init__(self, name, progress_bar=False):
        self.name = name
        self.progress_bar = progress_bar

    def get_sample(pool, sample_size, default_value):
        if sample_size == -1:
            return pool, None
        elif sample_size is not None and sample_size == pool.shape[0]:
            return pool, None
        elif sample_size is not None:
            sample, newpool = train_test_split(pool, train_size=sample_size)
            assert sample.shape[0] == sample_size
            assert sample.shape[0] + newpool.shape[0] == pool.shape[0]
            return sample, newpool
        else:
            return default_value, pool

    def __call__(self, x,
                 X0=None, X1=None,
                 train_size=None, calibrate_size=None, test_size=None,
                 class0_train_size=None, class0_calibrate_size=None, class0_test_size=None,
                 class1_train_size=None, class1_calibrate_size=None, class1_test_size=None,
                 class0_train=None, class1_train=None, class0_calibrate=None, class1_calibrate=None, class0_test=None, class1_test=None,
                 distribution_mean_delta=None,
                 train_folds=None, train_reuse=False, repeat=1):

        if distribution_mean_delta is not None:
            self._distribution_mean_delta = distribution_mean_delta

        resolve = lambda value: value if value is None or not callable(value) else value(x)

        cllr = []
        for run in tqdm(range(repeat), desc='{} ({})'.format(self.name, x), disable=not self.progress_bar):
            class0_pool = resolve(X0)
            class1_pool = resolve(X1)

            class0_train, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool, class0_train_size if class0_train_size is not None else train_size, resolve(class0_train))
            class0_calibrate, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool, class0_calibrate_size if class0_calibrate_size is not None else calibrate_size, resolve(class0_calibrate))
            class0_test, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool, class0_test_size if class0_test_size is not None else test_size, resolve(class0_test))

            class1_train, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool, class1_train_size if class1_train_size is not None else train_size, resolve(class1_train))
            class1_calibrate, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool, class1_calibrate_size if class1_calibrate_size is not None else calibrate_size, resolve(class1_calibrate))
            class1_test, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool, class1_test_size if class1_test_size is not None else test_size, resolve(class1_test))

            if class0_train is not None and train_folds is not None:
                LOG.debug('evaluate cllr kfold')
                assert class0_train is not None
                assert class1_train is not None
                assert class0_test is not None
                assert class1_test is not None
                cllr.append(self.cllr_kfold(train_folds, class0_train, class1_train, class0_test, class1_test))
            elif class0_calibrate is not None:
                LOG.debug('evaluate cllr')
                cllr.append(self.cllr(class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test))
            elif class0_train is not None and train_reuse:
                LOG.debug('evaluate cllr, reuse training set for calibration')
                cllr.append(self.cllr(class0_train, class1_train, class0_train, class1_train, class0_test, class1_test))

        return cllr


class NormalCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, loc0, scale0, loc1, scale1):
        super().__init__(name)

        self._loc0 = loc0
        self._scale0 = scale0
        self._loc1 = loc1
        self._scale1 = scale1
        self._distribution_mean_delta = None

    def _get_probability(X, mu, sigma):
        return np.exp(-np.power(X - mu, 2) / (2*sigma*sigma)) / math.sqrt(2*math.pi*sigma*sigma)

    def _get_lr(self, X):
        # calculate P(E|H0)
        X_p0 = NormalCllrEvaluator._get_probability(X, self._loc0, self._scale0)
        # calculate P(E|H1)
        X_p1 = NormalCllrEvaluator._get_probability(X, self._loc1, self._scale1)
        # calculate LR
        return X_p1 / X_p0

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        return self.calculate_cllr(X0_test, X1_test)

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        return self.calculate_cllr(class0_test, class1_test)

    def calculate_cllr(self, class0_test, class1_test):
        assert class0_test.shape[1] == 1

        # adjust loc1
        if self._distribution_mean_delta is not None:
            self._loc1 = self._loc0 + self._distribution_mean_delta

        # sample from H0
        X0_lr = self._get_lr(class0_test.reshape(-1))
        # sample from H1
        X1_lr = self._get_lr(class1_test.reshape(-1))

        cllr = lr.calculate_cllr(X0_lr, X1_lr)
        return cllr


class ScoreBasedCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, clf, density_function, preprocessors, progress_bar=False):
        super().__init__(name, progress_bar)

        self._clf = clf
        self._pfunc = density_function
        self._preprocessors = preprocessors

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        for p in self._preprocessors:
            X0_train, X1_train, X0_test, X1_test = process_vector(p, X0_train, X1_train, X0_test, X1_test)
        cllr = lr.scorebased_cllr_kfold(self._clf, self._pfunc, n_splits, X0_train, X1_train, X0_test, X1_test)
        return cllr

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        for p in self._preprocessors:
            class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test = process_vector(p, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test)
        cllr = lr.scorebased_cllr(self._clf, self._pfunc, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test)
        return cllr


class PlotCllrAvg:
    def ylabel():
        return 'C_llr'

    def value(cllr_lst):
        return sum([d.cllr for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return np.std([d.cllr for d in cllr_lst])


class PlotCllrStd:
    def ylabel():
        return 'std(C_llr)'

    def value(cllr_lst):
        return PlotCllrAvg.std(cllr_lst)

    def std(cllr_lst):
        return None


class PlotCllrCal:
    def ylabel():
        return 'C_llr calibration loss'

    def value(cllr_lst):
        return sum([d.cllr_cal for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return None


class PlotLlrAvg:
    def ylabel():
        return 'llr_h0'

    def value(cllr_lst):
        return sum([d.avg_llr_class0 for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return np.std([d.avg_llr_class0 for d in cllr_lst])


class PlotLlrStd:
    def ylabel():
        return 'std(llr_h0)'

    def value(cllr_lst):
        return PlotLlrAvg.std(cllr_lst)

    def std(cllr_lst):
        return None


def makeplot_density(clf, X0_train, X1_train, X0_calibrate, X1_calibrate, calibrators, savefig=None, show=None):
    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', ]

    plt.figure(figsize=(20,20), dpi=100)

    clf.fit_classifier(*Xn_to_Xy(X0_train, X1_train))
    points0 = lr.apply_scorer(clf, X0_calibrate)
    points1 = lr.apply_scorer(clf, X1_calibrate)

    for name, f in calibrators:
        f.fit_classifier(*Xn_to_Xy(points0, points1))

    x = np.arange(0, 1, .01)

    plt.hist(points0, bins=20, alpha=.25, density=True)
    plt.hist(points1, bins=20, alpha=.25, density=True)

    for i, nf in enumerate(calibrators):
        name, f = nf
        f.transform(x)
        plt.plot(x, f.p0, label=name, c=line_colors[i])
        plt.plot(x, f.p1, label=name, c=line_colors[i])

    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def makeplot_hist_density(lrs, y_nhot, calibrators, target_classes, label_encoder, savefig=None, show=None):
    """
    Makes histogram for calibration data for one hypothesis and plots the KDE curve.

    :param lrs:
    :param y_nhot:
    :param calibrators:
    :param target_classes:
    :param savefig: boolean if True the figure is saved
    :param show: boolean if True the figure is displayed
    :return:
    """

    loglrs = np.log10(lrs)

    plt.subplots(2, int(loglrs.shape[1]/2), figsize=(14, 9))
    for i, target_class in enumerate(target_classes):
        celltype = vec2string(target_class, label_encoder)

        loglrs1 = np.multiply(loglrs[:, i], np.max(np.multiply(y_nhot, target_class), axis=1))
        loglrs2 = np.multiply(loglrs[:, i], 1 - np.max(np.multiply(y_nhot, target_class), axis=1))

        loglrs1 = np.delete(loglrs1, np.where(loglrs1 == -0.0))
        loglrs2 = np.delete(loglrs2, np.where(loglrs2 == 0.0))

        X = np.arange(np.min(np.append(loglrs1, loglrs2))-0.25,
                      np.max(np.append(loglrs1, loglrs2))+0.25, .01)

        X = X.reshape(-1, 1)
        calibrators[str(target_class)].transform(X)

        plt.subplot(2, int(lrs.shape[1] / 2), i + 1)
        plt.hist(loglrs1, density=True, color='orange', label='h1', bins=30, alpha=0.5)
        plt.hist(loglrs2, density=True, color='blue', label='h2', bins=30, alpha=0.5)

        plt.plot(X, calibrators[str(target_class)].p1, color='orange', label='density h1')
        plt.plot(X, calibrators[str(target_class)].p0, color='blue', label='density h2')

        plt.title(celltype)
        plt.xlabel('10logLR')
        plt.ylabel('Density')
        plt.legend(loc=2, framealpha=0.5)

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

    plt.close()

# TODO: Change this function
def makeplot_hist_density_avg(probs, calibrators, savefig=None, show=None):

    X = np.arange(-0.25, 1.25, .01)
    X = X.reshape(-1, 1)

    plt.subplots(2, int(len(probs)/2), figsize=(14, 9))
    for idx, (celltype, list_probs) in enumerate(sorted(probs.items())):
        probs0 = np.array(list_probs[0])
        probs1 = np.array(list_probs[1])

        plt.subplot(2, int(len(probs)/2), idx + 1)
        plt.hist(probs0, density=True, color='orange', label='avg h0', bins=30, alpha=0.5)
        plt.hist(probs1, density=True, color='blue', label='avg h1', bins=30, alpha=0.5)

        p0s = []
        p1s = []
        for calibrator in calibrators[celltype]:
            calibrator.transform(X)
            p0s.append(calibrator.p0)
            p1s.append(calibrator.p1)

        avgp0 = np.mean(np.array([p0 for p0 in p0s]), axis=0)
        avgp1 = np.mean(np.array([p1 for p1 in p1s]), axis=0)

        plt.plot(X, avgp0, color='black', linestyle='--', label='avg density h0')
        plt.plot(X, avgp1, color='black', linestyle=':', label='avg density h1')
        plt.title(celltype)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend(loc=2, framealpha=0.5)

        # if idx == 0:
        #     plt.legend(loc=2, framealpha=0.5)

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

# TODO: Change this function
def makeplot_density_avg(calibrators, savefig=None, show=None):
    linestyles = ['-', '--', '-.', ':', ]

    X = np.arange(-0.25, 1.25, .01)
    X = X.reshape(-1, 1)

    plt.subplots(2, int(len(calibrators)/2), figsize=(14, 9))
    for idx, (celltype, list_calibs) in enumerate(sorted(calibrators.items())):

        plt.subplot(2, int(len(calibrators)/2), idx + 1)
        p0s = []
        p1s = []
        for i, calibrator in enumerate(list_calibs):

            calibrator.transform(X)

            plt.plot(X, calibrator.p0, color='blue', linestyle=linestyles[i])
            plt.plot(X, calibrator.p1, color='orange', linestyle=linestyles[i])

            p0s.append(calibrator.p0)
            p1s.append(calibrator.p1)

        avgp0 = np.mean(np.array([p0 for p0 in p0s]), axis=0)
        avgp1 = np.mean(np.array([p1 for p1 in p1s]), axis=0)

        plt.plot(X, avgp0, color='black', linewidth=3)
        plt.plot(X, avgp1, color='black', linewidth=3)
        plt.title(celltype)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        # plt.legend(loc=2, framealpha=0.5)

        # if idx == 0:
        #     plt.legend(loc=2, framealpha=0.5)

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def makeplot_cllr(xlabel, generators, experiments, savefig=None, show=None, plots=[PlotCllrAvg, PlotCllrStd, PlotCllrCal]):
    plt.figure(figsize=(20,20), dpi=100)

    axs = None

    xvalues, _ = zip(*experiments)

    for g in generators:
        LOG.debug('makeplot_cllr: {name}'.format(name=g.name))
        stats = [ g(x=x, **genargs) for x, genargs in experiments ]

        if axs is None:
            axs = []
            for i, plot in enumerate(plots):
                ax = plt.subplot(len(plots), 1, i+1)
                plt.ylabel(plot.ylabel())
                axs.append(ax)

            plt.xlabel(xlabel)

        for i in range(len(plots)):
            plot = plots[i]
            ax = axs[i]
            axplot = ax.plot(xvalues, [ plot.value(d) for d in stats ], 'o--', label=g.name)[0]
            if plot.std(stats[0]) is not None:
                ax.plot(xvalues, [ (plot.value(d)-plot.std(d), plot.value(d)+plot.std(d)) for d in stats ], '_', color=axplot.get_color())

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1), ncol=2)

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def makeplot_accuracy(scorer, density_function, X0_train, X1_train, X0_calibrate, X1_calibrate, title, labels=('class0', 'class1'), savefig=None, show=None):
    LOG.debug('makeplot_accuracy')
    stats = lr.scorebased_cllr(scorer, density_function, X0_train, X1_train, X0_calibrate, X1_calibrate)

    scale = 2

    plt.figure(figsize=(20,20), dpi=100)

    bins0 = collections.defaultdict(float)
    for v in stats.lr_class0:
        bins0[int(round(math.log(v, scale)))] += (1 if v < 1 else v) / len(stats.lr_class0)

    bins1 = collections.defaultdict(float)
    for v in stats.lr_class1:
        bins1[int(round(math.log(v, scale)))] += (1 if v > 1 else 1/v) / len(stats.lr_class1)

    bins0_x, bins0_y = zip(*sorted(bins0.items()))
    bins1_x, bins1_y = zip(*sorted(bins1.items()))

    plt.bar(np.array(bins0_x) - .15, bins0_y, label=labels[0], width=.3)
    plt.bar(np.array(bins1_x) + .15, bins1_y, label=labels[1], width=.3)

    plt.title(title)

    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()
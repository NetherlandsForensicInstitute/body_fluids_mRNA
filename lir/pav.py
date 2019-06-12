import matplotlib.pyplot as plt
import numpy as np
import sklearn.isotonic


def plot(lrs, y, show_scatter=True, on_screen=False, path=None, kw_figure={}):
    llrs = np.log10(lrs)

    pav = PavLogLR()
    pav_llrs = pav.fit_transform(llrs, y)

    all_llrs = np.concatenate([llrs, pav_llrs])
    all_llrs[all_llrs == -np.inf] = 0
    all_llrs[all_llrs == np.inf] = 0
    xrange = [all_llrs.min() - .5, all_llrs.max() + .5]

    fig = plt.figure(**kw_figure)
    plt.axis(xrange + xrange)
    plt.plot(xrange, xrange)  # rechte lijn door de oorsprong

    pav_x = np.arange(*xrange, .01)
    plt.plot(pav_x, pav.transform(pav_x))  # pre-/post-calibrated lr fit

    if show_scatter:
        plt.scatter(llrs, pav_llrs)  # scatter plot of measured lrs

    plt.xlabel("pre-calibrated 10log(lr)")
    plt.ylabel("post-calibrated 10log(lr)")
    plt.grid(True, linestyle=':')
    if on_screen:
        plt.show()
    if path is not None:
        plt.savefig(path)

    plt.close(fig)


class PavLR:
    """
    Isotonic regression model applied on LRs.
    """

    ir = sklearn.isotonic.IsotonicRegression()

    def __init__(self, prior=None):
        self.prior = prior

    def fit(self, X, y):
        """
        Arguments:
        X: an array of LRs
        y: an array of labels (values 0 or 1)
        """
        self.ir.fit(self.to_probability(X), y, sample_weight=self.to_weight(y))

    def transform(self, X):
        """
        Arguments:
        X: an array of LRs
        """
        pav_p = self.ir.transform(self.to_probability(X))
        return pav_p / (1 - pav_p)

    def fit_transform(self, X, y):
        pav_p = self.ir.fit_transform(self.to_probability(X), y, sample_weight=self.to_weight(y))
        with np.errstate(divide='ignore'):
            return pav_p / (1 - pav_p)

    def to_probability(self, lr):
        return lr / (1 + lr)

    def to_lr(self, p):
        with np.errstate(divide='ignore'):
            return p / (1 - p)

    def to_weight(self, y):
        prior = self.prior if self.prior is not None else np.sum(y) / y.size
        return (1 - y) * prior + y * (1 - prior)


class PavLogLR(PavLR):
    def __init__(self, prior=None):
        super().__init__(prior)

    def fit(self, X, y):
        """
        Arguments:
        X: an array of log(LR)
        y: an array of labels (values 0 or 1)
        """
        return super().fit(np.exp(X), y)

    def transform(self, X):
        """
        Arguments:
        X: an array of log(LR)
        """
        with np.errstate(divide='ignore'):
            return np.log(super().transform(np.exp(X)))

    def fit_transform(self, X, y):
        with np.errstate(divide='ignore'):
            return np.log(super().fit_transform(np.exp(X), y))
import numpy as np
import matplotlib.pyplot as plt

from rna.utils import vec2string

from lir.calibration import IsotonicCalibrator

def plot_ece(lrs, y_nhot, target_classes, label_encoder):

    n_target_classes = len(target_classes)

    priors = np.linspace(0.001, 1-0.001, 50).tolist()

    plt.subplots(int(n_target_classes / 2), 2, figsize=(9, int(9 / 4 * n_target_classes)), sharey='row')
    for i, target_class in enumerate(target_classes):

        celltype = vec2string(target_class, label_encoder)

        lrs_p = np.multiply(lrs[:, i], np.max(np.multiply(y_nhot, target_class), axis=1))
        lrs_d = np.multiply(lrs[:, i], 1-np.max(np.multiply(y_nhot, target_class), axis=1))

        # delete zeros
        lrs_p = np.delete(lrs_p, np.where(lrs_p == -0.0))
        lrs_d = np.delete(lrs_d, np.where(lrs_d == 0.0))

        # LR = 1
        results_LR_1 = np.array([emperical_cross_entropy(np.ones_like(lrs_p), np.ones_like(lrs_d), prior) for prior in priors])
        ece_LR_1 = results_LR_1[:, 0]
        odds = results_LR_1[:, 1]

        # True LR
        results = np.array([emperical_cross_entropy(lrs_p, lrs_d, prior) for prior in priors])
        ece = results[:, 0]

        # LR after calibration
        lrs_after_calibration_p, lrs_after_calibration_d = transform_lrs(lrs[:, i], y_nhot, target_class)
        results_after_calib = np.array([emperical_cross_entropy(lrs_after_calibration_p, lrs_after_calibration_d, prior) for prior in priors])
        ece_after_calib = results_after_calib[:, 0]

        plt.subplot(int(n_target_classes / 2), 2, i + 1)
        plt.plot(np.log10(odds), ece_LR_1, label='LR=1 always')
        plt.plot(np.log10(odds), ece, label='LR values')
        plt.plot(np.log10(odds), ece_after_calib, label='LR after PAV')

        plt.title(celltype)
        plt.legend()
        plt.ylabel("Emperical Cross-Entropy")
        plt.xlabel("Prior log10 (odds)")


def emperical_cross_entropy(LR_p, LR_d, prior):
    """

    :param LR_p: LRs with ground truth label prosecution
    :param prior_p: int; fixed value for prior prosecution
    :param LR_d: LRs with groudn trut label defence
    :param prior_d: int; fixed value for prior defence
    :return:
    """

    N_p = len(LR_p)
    N_d = len(LR_d)

    prior_p = prior
    prior_d = 1 - prior
    odds = prior_p / prior_d

    return (prior_p / N_p) * np.sum(np.log2(1 + (1/(LR_p * odds)))) + \
            (prior_d / N_d) * np.sum(np.log2(1 + (LR_d * odds))), odds


def transform_lrs(lrs, y_nhot, target_class):

    y = np.max(np.multiply(y_nhot, target_class), axis=1)

    ir = IsotonicCalibrator()
    ir.fit(lrs, y)
    lrs_after_calibration = ir.transform(lrs)

    lrs_after_calibration = np.where(lrs_after_calibration > 10 ** 10, 10 ** 10, lrs_after_calibration)
    lrs_after_calibration = np.where(lrs_after_calibration < 10 ** -10, 10 ** -10, lrs_after_calibration)

    lrs_after_calibration_p = lrs_after_calibration[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 1)]
    lrs_after_calibration_d = lrs_after_calibration[np.argwhere(np.max(np.multiply(y_nhot, target_class), axis=1) == 0)]

    return lrs_after_calibration_p, lrs_after_calibration_d

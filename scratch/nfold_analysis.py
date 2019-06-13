"""
Run nfold analysis.
"""

# import settings
import time
import os

import numpy as np

from sklearn.model_selection import train_test_split

from rna.analytics import augment_data, cllr, combine_samples
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, MarginalXGBClassifier
from rna.utils import vec2string, string2vec, MultiLabelEncoder
from rna.plotting import plot_histogram_log_lr, plot_boxplot_of_metric

from scratch.single_analysis import get_accuracy

AUGMENT = [True]
BINARIZE = [True, False]
MARKERS = [True, False]
LPS = [True, False]
MODELS = ['MLP', 'MLR', 'XGB']

# def all_combinations()

def nfold_analysis(nfolds, tc, augment, binarize, markers, lps, nsamples, test_size, calibration_size, model, with_calibration=True):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = get_data_per_cell_type(
        single_cell_types=single_cell_types, markers=markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)
    X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=markers)

    # ======= Initialize =======
    # TODO: Find nice way to save metrics
    train_accuracy, test_accuracy, mixture_accuracy, single_accuracy, cllr_menstr, cllr_menstr_mixt, cllr_nasal, \
    cllr_nasal_mixt, cllr_saliva, cllr_saliva_mixt, cllr_skin, cllr_skin_mixt, cllr_vag, cllr_vag_mixt, cllr_vag_menstr, \
    cllr_vag_menstr_mixt = (np.zeros((nfolds, len(BINARIZE), len(LPS), len(MODELS))) for i in range(16))

    for n in range(nfolds):
        # ======= Split data & Augment data =======
        if with_calibration:
            X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=test_size)
            X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=calibration_size)

            X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features, nsamples[0], label_encoder, binarize=binarize, from_penile=from_penile)
            X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features, nsamples[1], label_encoder, binarize=binarize, from_penile=from_penile)
            X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features, nsamples[2], label_encoder, binarize=binarize, from_penile=from_penile)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=test_size)

            X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features, nsamples[0], label_encoder, binarize=binarize, from_penile=from_penile)
            X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features, nsamples[2], label_encoder, binarize=binarize, from_penile=from_penile)
            X_calib_augmented, y_calib_nhot_augmented = [np.array([]) for i in range (2)]

        X_test = transform_test_data(X_test, binarize)

        model = model_with_correct_settings(model, lps)

        # ======= Calculate LRs before and after calibration =======
        lrs_before_calib, lrs_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = generate_lrs(model, mle, lps, X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented, X_test_augmented, X_mixtures, target_classes)

        plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, savefig=os.path.join('scratch, hist_before_{}_{}_{}_{}'.format(n, binarize, probability_calculation, MODEL)))
        plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder, density=True, title='after', savefig=os.path.join('scratch, hist_after_{}_{}_{}_{}'.format(n, binarize, probability_calculation, MODEL)))

        # ======= Calculate accuracy and Cllr =======
        # TODO: Find nice way to save metrics
        train_accuracy[n, i, j, k] = get_accuracy(model, mle, y_train_nhot_augmented, X_train_augmented, target_classes)
        test_accuracy[n, i, j, k] = get_accuracy(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)
        mixture_accuracy[n, i, j, k] = get_accuracy(model, mle, y_nhot_mixtures, X_mixtures, target_classes)
        if binarize == True:
            single_accuracy[n, i, j, k] = get_accuracy(model, mle, mle.inv_transform_single(y_test), X_test_bin, target_classes)
        else:
            single_accuracy[n, i, j, k] = get_accuracy(model, mle, mle.inv_transform_single(y_test), X_test_norm, target_classes)

        cllr_menstr[n, i, j, k] = cllr(lrs_after_calib[:, 0], y_test_nhot_augmented, target_classes[0])
        cllr_menstr_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 0], y_nhot_mixtures, target_classes[0])
        cllr_nasal[n, i, j, k] = cllr(lrs_after_calib[:, 1], y_test_nhot_augmented, target_classes[1])
        cllr_nasal_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 1], y_nhot_mixtures, target_classes[1])
        cllr_saliva[n, i, j, k] = cllr(lrs_after_calib[:, 2], y_test_nhot_augmented, target_classes[2])
        cllr_saliva_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 2], y_nhot_mixtures, target_classes[2])
        cllr_skin[n, i, j, k] = cllr(lrs_after_calib[:, 3], y_test_nhot_augmented, target_classes[3])
        cllr_skin_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 3], y_nhot_mixtures, target_classes[3])
        cllr_vag[n, i, j, k] = cllr(lrs_after_calib[:, 4], y_test_nhot_augmented, target_classes[4])
        cllr_vag_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 4], y_nhot_mixtures, target_classes[4])
        cllr_vag_menstr[n, i, j, k] = cllr(lrs_after_calib[:, 5], y_test_nhot_augmented, target_classes[5])
        cllr_vag_menstr_mixt[n, i, j, k] = cllr(lrs_after_calib_mixt[:, 5], y_nhot_mixtures, target_classes[5])

        end = time.time()
        print("Execution time fold {} in seconds: {}".format(n, (end - start)))

    print("\nMean of (augmented) train accuracy:")
    print("------------------------------------")
    print(np.mean(train_accuracy, axis=0)[:, :, :])

    print("\nMean of (augmented) test accuracy:")
    print("------------------------------------")
    print(np.mean(test_accuracy, axis=0)[:][:][:])

    print("\nMean of (original) mixture accuracy:")
    print("------------------------------------")
    print(np.mean(mixture_accuracy, axis=0)[:][:][:])

    print("\nMean of (original) single accuracy:")
    print("------------------------------------")
    print(np.mean(single_accuracy, axis=0)[:][:][:])

    print("\nMean of Cllr_vag_menstr:")
    print("------------------------------------")
    print(np.mean(cllr_vag_menstr, axis=0)[:][:][:])

    print("\nMean of Cllr_vag_menstr_mixt:")
    print("------------------------------------")
    print(np.mean(cllr_vag_menstr_mixt, axis=0)[:][:][:])

    plot_boxplot_of_metric(train_accuracy, "train accuracy", savefig='scratch/boxplot_train_accuracy')
    plot_boxplot_of_metric(test_accuracy, "test accuracy", savefig='scratch/boxplot_test_accuracy')
    plot_boxplot_of_metric(mixture_accuracy, "mixture accuracy", savefig='scratch/boxplot_mixture_accuracy')
    plot_boxplot_of_metric(single_accuracy, "single accuracy", savefig='scratch/boxplot_single_accuracy')

    plot_boxplot_of_metric(cllr_menstr, "cllr menstr", savefig='scratch/boxplot_cllr_menstr')
    plot_boxplot_of_metric(cllr_menstr_mixt, "cllr menstr mixt", savefig='scratch/boxplot_cllr_menstr_mixt')
    plot_boxplot_of_metric(cllr_nasal, "cllr nasal", savefig='scratch/boxplot_cllr_nasal')
    plot_boxplot_of_metric(cllr_nasal_mixt, "cllr nasal mixt", savefig='scratch/boxplot_cllr_nasal_mixt')
    plot_boxplot_of_metric(cllr_saliva, "cllr saliva", savefig='scratch/boxplot_cllr_saliva')
    plot_boxplot_of_metric(cllr_saliva_mixt, "cllr saliva mixt", savefig='scratch/boxplot_cllr_saliva_mixt')
    plot_boxplot_of_metric(cllr_skin, "cllr skin", savefig='scratch/boxplot_cllr_skin')
    plot_boxplot_of_metric(cllr_skin_mixt, "cllr skin mixt", savefig='scratch/boxplot_cllr_skin_mixt')
    plot_boxplot_of_metric(cllr_vag, "cllr vag", savefig='scratch/boxplot_cllr_vag')
    plot_boxplot_of_metric(cllr_vag_mixt, "cllr vag_mixt", savefig='scratch/boxplot_cllr_vag_mixt')
    plot_boxplot_of_metric(cllr_vag_menstr, "cllr vag menstr", savefig='scratch/boxplot_cllr_vag_menstr')
    plot_boxplot_of_metric(cllr_vag_menstr_mixt, "cllr vag menstr mixt", savefig='scratch/boxplot_cllr_vag_menstr_mixt')

    print("END")


def transform_test_data(X_test, binarize):
    X_test = combine_samples(X_test)
    if binarize:
        X_test = np.where(X_test > 150, 1, 0)
    else:
        X_test = X_test / 1000
    return X_test

def model_with_correct_settings(model_no_settings, probability_calculation_softmax):
    """
    Ensures that the correct model with correct settings is used in the analysis.
    This is based on a string 'model_no_settings' and a boolean deciding how the
    probabilties are calculated 'probabilitye_calculation': either with the softmax
    function or the sigmoid function.

    :param model_no_settings: str: model
    :param probability_calculation: boolean: if True the softmax function is used to
        calculate the probabilities with.
    :return: model with correct settings
    """

    if model_no_settings == 'MLP' and probability_calculation_softmax: # softmax
        model = MarginalMLPClassifier()
    elif model_no_settings == 'MLP' and not probability_calculation_softmax: # sigmoid
        model = MarginalMLPClassifier(activation='logistic')

    elif model_no_settings == 'MLR' and probability_calculation_softmax:
        model = MarginalMLRClassifier(multi_class='multinomial', solver='newton-cg')
    elif model_no_settings == 'MLR' and not probability_calculation_softmax:
        model = MarginalMLRClassifier()

    elif model_no_settings == 'XGB' and probability_calculation_softmax:
        model = MarginalXGBClassifier()
    elif model_no_settings == 'XGB' and not probability_calculation_softmax:
        model = MarginalXGBClassifier(method='sigmoid')

    else:
        raise ValueError("")

    return model


def generate_lrs(model, mle, method, X_train, y_train, X_calib, y_calib, X_test, X_mixtures, target_classes):
    """
    When softmax the model must be fitted on labels, whereas with sigmoid the model must be fitted on
    an nhot encoded vector representing the labels. Ensure that labels take the correct form, fit the
    model and predict the lrs before and after calibration for both X_test and X_mixtures.
    """

    if method: # y_train must be list with labels
        try:
            y_train = mle.nhot_to_labels(y_train)
        except: # already are labels
            pass
    else: # y_train must be nhot encoded labels
        try:
            y_train = mle.labels_to_nhot(y_train)
        except: # already is nhot encoded
            pass
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(y_train[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    try: # y_calib must always be nhot encoded
        y_calib = mle.labels_to_nhot(y_calib)
    except: # already is nhot encoded
        pass

    model.fit_classifier(X_train, y_train)
    model.fit_calibration(X_calib, y_calib, target_classes)

    lrs_before_calib = model.predict_lrs(X_test, target_classes, with_calibration=False)
    lrs_after_calib = model.predict_lrs(X_test, target_classes)

    lrs_before_calib_mixt = model.predict_lrs(X_mixtures, target_classes, with_calibration=False)
    lrs_after_calib_mixt = model.predict_lrs(X_mixtures, target_classes)

    return lrs_before_calib, lrs_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt





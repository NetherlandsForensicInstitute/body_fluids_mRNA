"""
Run nfold analysis.
"""

import settings

import numpy as np

from sklearn.model_selection import train_test_split

from rna.analytics import augment_data, cllr, combine_samples
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, MarginalXGBClassifier
from rna.utils import vec2string, string2vec, MultiLabelEncoder
from rna.plotting import plot_histogram_log_lr, plot_boxplot_of_metric

from scratch.single_analysis import get_accuracy

# from lir.plotting import makeplot_hist_density

BINARIZE = [True, False]
LPS = [True, False]
MODELS = ['MLP', 'MLR', 'XGB']
# MODELS = ['MLR']

def nfold_analysis(nfolds, tc):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = get_data_per_cell_type(
        single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    # ======= Initialize =======
    train_accuracy, test_accuracy, mixture_accuracy, single_accuracy, cllr_menstr, cllr_menstr_mixt, cllr_nasal, \
    cllr_nasal_mixt, cllr_saliva, cllr_saliva_mixt, cllr_skin, cllr_skin_mixt, cllr_vag, cllr_vag_mixt, cllr_vag_menstr, \
    cllr_vag_menstr_mixt = (np.zeros((nfolds, len(BINARIZE), len(LPS), len(MODELS))) for i in range(16))

    for n in range(nfolds):
        print("Fold {}".format(n))

        # ======= Split data =======
        X_train_mlr, X_test, y_train_mlr, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train_mlr, y_train_mlr, stratify=y_train_mlr, test_size=settings.calibration_size)

        for i, binarize in enumerate(BINARIZE):
            print("Binarize: {}".format(binarize))
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)

            # ======= Augment data =======
            X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features, settings.nsamples[0], label_encoder, binarize=binarize, from_penile=from_penile)
            X_train_augmented_mlr, y_train_nhot_augmented_mlr = augment_data(X_train_mlr, y_train_mlr, n_celltypes, n_features, settings.nsamples[0] + settings.nsamples[1], label_encoder, binarize=binarize, from_penile=from_penile)
            X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features, settings.nsamples[1], label_encoder, binarize=binarize, from_penile=from_penile)
            X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features, settings.nsamples[2], label_encoder, binarize=binarize, from_penile=from_penile)

            # ======= Convert data accordingly =======
            X_test_comb = combine_samples(X_test)
            X_test_bin = np.where(X_test_comb > 150, 1, 0)
            X_test_norm = X_test_comb / 1000

            for j, method in enumerate(LPS):
                print("LPS: {}".format(method))

                for k, MODEL in enumerate(MODELS):
                    print("MODEL: {}".format(MODEL))
                    model = model_with_correct_settings(MODEL, method)

                    # ======= Calculate LRs before and after calibration =======
                    if MODEL != 'MLR':
                        lrs_before_calib, lrs_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                            generate_lrs(model, mle, method, X_train_augmented, y_train_nhot_augmented, X_calib_augmented,
                                         y_calib_nhot_augmented, X_test_augmented, X_mixtures, target_classes)

                    else:
                        X_calib_augmented_mlr = np.array([])
                        y_calib_nhot_augmented_mlr = np.array([])
                        lrs_before_calib, lrs_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                            generate_lrs(model, mle, method, X_train_augmented_mlr, y_train_nhot_augmented_mlr, X_calib_augmented_mlr,
                                         y_calib_nhot_augmented_mlr, X_test_augmented, X_mixtures, target_classes)

                    plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, savefig='scratch/hist_before_{}_{}_{}_{}'.format(n, binarize, method, MODEL))
                    plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder, density=True, title='after', savefig='scratch/hist_after_{}_{}_{}_{}'.format(n, binarize, method, MODEL))

                    # ======= Calculate accuracy and Cllr =======
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


def model_with_correct_settings(model_no_settings, method):

    if model_no_settings == 'MLP' and method: # softmax
        model = MarginalMLPClassifier()
    elif model_no_settings == 'MLP' and not method: # sigmoid
        model = MarginalMLPClassifier(activation='logistic')

    elif model_no_settings == 'MLR' and method:
        model = MarginalMLRClassifier(multi_class='multinomial', solver='newton-cg')
    elif model_no_settings == 'MLR' and not method:
        model = MarginalMLRClassifier()

    elif model_no_settings == 'XGB' and method:
        model = MarginalXGBClassifier()
    elif model_no_settings == 'XGB' and not method:
        model = MarginalXGBClassifier(method='sigmoid')

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





"""
Performs analysis
"""

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rna.analytics import augment_data, cllr, combine_samples
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalMLPClassifier, MarginalMLRClassifier, MarginalXGBClassifier
from rna.utils import vec2string, string2vec, MultiLabelEncoder
from rna.plotting import plot_histogram_log_lr
from scratch.plotting import plot_ece

import rna.settings as settings

from lir.plotting import makeplot_hist_density

output_activation = {True:'relu',
                     False:'logistic'}

softmax_sigmoid_MLR = {True: ['multinomial', 'newton-cg'],
                       False: ['ovr', 'liblinear']}

softmax_sigmoid_XGB = {True: 'softmax',
                       False: 'sigmoid'}

def individual_analysis(tc, show=True, treat_replicates_as_single=False):

    if settings.lps and not settings.augment:
        raise ValueError("Cannot use label powerset method with the original data.")

    if settings.model == 'MLR' and settings.calibration_size != 0.0:
        raise ValueError("Cannot calibrate with multinominal logistic regressions, "
                         "either set calibration_size=0.0 or use another model.")

    from_penile=False # TODO: Currently doesn't do anything
    mle = MultiLabelEncoder(len(single_cell_types))
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = get_data_per_cell_type(
        single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)
    X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=settings.binarize, markers=settings.markers)

    if treat_replicates_as_single:
        X_single_nrp, y_nhot_single_nrp, y_single_nrp = use_repeated_measurements_as_single(X_single, y_nhot_single, y_single)

    assert X_single[0].shape[1] == X_mixtures.shape[1]

    if treat_replicates_as_single:
        X_train_nrp, X_test_nrp, y_train_nrp, y_test_nrp = train_test_split(X_single_nrp, y_single_nrp, stratify=y_single_nrp, test_size=settings.test_size)
    X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)

    if settings.calibration_size == 0.0:
        if treat_replicates_as_single:
            X_calib_nrp = np.array([])
            y_calib_nrp = np.array([])
        else:
            X_calib = np.array([])
            y_calib = np.array([])
    else:
        if treat_replicates_as_single:
            X_train_nrp, X_calib_nrp, y_train_nrp, y_calib_nrp = train_test_split(X_train_nrp, y_train_nrp, stratify=y_train_nrp, test_size=settings.calibration_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

    X_train_augmented, y_train_nhot_augmented = augment_data(X_train, y_train, n_celltypes, n_features, settings.nsamples[0], label_encoder, binarize=settings.binarize, from_penile=from_penile)
    X_calib_augmented, y_calib_nhot_augmented = augment_data(X_calib, y_calib, n_celltypes, n_features, settings.nsamples[1], label_encoder, binarize=settings.binarize, from_penile=from_penile)
    X_test_augmented, y_test_nhot_augmented = augment_data(X_test, y_test, n_celltypes, n_features, settings.nsamples[2], label_encoder, binarize=settings.binarize, from_penile=from_penile)
    X_test_augmented_reduced, y_test_nhot_augmented_reduced = only_use_same_combinations_as_in_mixtures(X_test_augmented, y_test_nhot_augmented, y_nhot_mixtures)

    X_train = combine_samples(X_train)
    X_calib = combine_samples(X_calib)
    X_test = combine_samples(X_test)
    if settings.binarize: # binarize
        if treat_replicates_as_single:
            X_train_nrp = np.where(X_train_nrp > 150, 1, 0)
            X_calib_nrp = np.where(X_calib_nrp > 150, 1, 0)
            X_test_nrp = np.where(X_test_nrp > 150, 1, 0)
        else:
            X_train = np.where(X_train > 150, 1, 0)
            X_calib = np.where(X_calib > 150, 1, 0)
            X_test = np.where(X_test > 150, 1, 0)

    else: # normalize
        if treat_replicates_as_single:
            X_train_nrp = X_train_nrp / 1000
            X_calib_nrp = X_calib_nrp / 1000
            X_test_nrp = X_test_nrp / 1000
        else:
            X_train = X_train / 1000
            X_calib = X_calib / 1000
            X_test = X_test / 1000

    print("\n===Settings===\n")
    total = sum([X_train.shape[0], X_calib.shape[0], X_test.shape[0]])
    print("true size (train, calib, test): ({0:.3f}, {1:.3f}, {2:.3f})".format(X_train.shape[0]/total,
                                                                               X_calib.shape[0]/total,
                                                                               X_test.shape[0]/total))
    print_settings(settings)

    if settings.model == 'MLP':
        model = MarginalMLPClassifier(activation=output_activation[settings.lps])
    elif settings.model == 'MLR':
        model = MarginalMLRClassifier(multi_class=softmax_sigmoid_MLR[settings.lps][0], solver=softmax_sigmoid_MLR[settings.lps][1])
    elif settings.model == 'XGB':
        model = MarginalXGBClassifier(method=softmax_sigmoid_XGB[settings.lps])

    if settings.augment:
        print("\n===Training process===\n")
        print("Trains on {} samples".format(X_train_augmented.shape[0]))

        # use all augmented test samples
        lrs_before_calib, lrs_after_calib, lrs_reduced_before_calib, lrs_reduced_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(model, mle, X_train_augmented, y_train_nhot_augmented, X_calib_augmented,
                         y_calib_nhot_augmented, X_test_augmented, X_test_augmented_reduced, X_mixtures, target_classes)

    else:
        y_train = mle.inv_transform_single(y_train)
        if treat_replicates_as_single:
            y_train_nrp = y_train_nrp.reshape(-1, 1)
            y_train_nrp = mle.inv_transform_single(y_train_nrp)
        try:
            y_calib = mle.inv_transform_single(y_calib)
            if treat_replicates_as_single:
                y_calib_nrp = y_calib_nrp.reshape(-1, 1)
                y_calib_nrp = mle.inv_transform_single(y_calib_nrp)
        except:
            pass
        print("\n===Training process===\n")
        print("Trains on {} samples".format(X_train.shape[0]))
        lrs_before_calib, lrs_after_calib, lrs_reduced_before_calib, lrs_reduced_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
            generate_lrs(model, mle, X_train, y_train, X_calib, y_calib, X_test_augmented, X_test_augmented_reduced, X_mixtures, target_classes)

        if treat_replicates_as_single:
            lrs_before_calib, lrs_after_calib, lrs_reduced_before_calib, lrs_reduced_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                generate_lrs(model, mle, X_train_nrp, y_train_nrp, X_calib_nrp, y_calib_nrp, X_test_augmented, X_test_augmented_reduced, X_mixtures, target_classes)

    print("\n===Performance metrics===\n")
    if settings.augment:
        print("Train accuracy (augmented): {0:.4f}".format(get_accuracy(model, mle, y_train_nhot_augmented, X_train_augmented, target_classes)))
        print("Test accuracy (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)))
        print("Test accuracy reduced (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented_reduced, X_test_augmented_reduced, target_classes)))
        print("Mixture accuracy (original): {0:.4f}".format(get_accuracy(model, mle, y_nhot_mixtures, X_mixtures, target_classes)))
        print("Single accuracy (original): {0:.4f}\n".format(get_accuracy(model, mle, mle.inv_transform_single(y_test), X_test, target_classes)))
    else:
        if treat_replicates_as_single:
            print("Single train accuracy nrp (original): {0:.4f}".format(get_accuracy(model, mle, y_train_nrp, X_train_nrp, target_classes)))
            print("Test accuracy (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)))
            print("Test accuracy reduced (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented_reduced, X_test_augmented_reduced, target_classes)))
            print("Mixture accuracy (original): {0:.4f}".format(get_accuracy(model, mle, y_nhot_mixtures, X_mixtures, target_classes)))
            y_test_nrp = y_test_nrp.reshape(-1, 1)
            print("Single test accuracy nrp (original): {0:.4f}\n".format(get_accuracy(model, mle, mle.inv_transform_single(y_test_nrp), X_test_nrp, target_classes)))
        else:
            print("Single train accuracy (original): {0:.4f}".format(get_accuracy(model, mle, y_train, X_train, target_classes)))
            print("Test accuracy (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented, X_test_augmented, target_classes)))
            print("Test accuracy reduced (augmented): {0:.4f}".format(get_accuracy(model, mle, y_test_nhot_augmented_reduced, X_test_augmented_reduced, target_classes)))
            print("Mixture accuracy (original): {0:.4f}".format(get_accuracy(model, mle, y_nhot_mixtures, X_mixtures, target_classes)))
            print("Single test accuracy (original): {0:.4f}\n".format(get_accuracy(model, mle, mle.inv_transform_single(y_test), X_test, target_classes)))


    print("Calculated with augmented test data:")
    print("------------------------------------")
    for i, target_class in enumerate(target_classes):
        print("Cllr for {}: {}".format(vec2string(target_class, label_encoder),
                                       round(cllr(lrs_after_calib[:, i], y_test_nhot_augmented, target_class), 4)))
        print("     min/max LR before: {0:.5f} / {1:.5f}".format(lrs_before_calib[:, i].min(), lrs_before_calib[:, i].max()))
        print("     min/max LR after: {0:.5f} / {1:.5f}".format(lrs_after_calib[:, i].min(), lrs_after_calib[:, i].max()))


    print("\nCalculated with reduced augmented test data:")
    print("------------------------------------")
    for i, target_class in enumerate(target_classes):
        print("Cllr for {}: {}".format(vec2string(target_class, label_encoder),
                                       round(cllr(lrs_reduced_after_calib[:, i], y_test_nhot_augmented_reduced, target_class), 4)))
        print("     min/max LR before: {0:.5f} / {1:.5f}".format(lrs_reduced_before_calib[:, i].min(), lrs_reduced_before_calib[:, i].max()))
        print("     min/max LR after: {0:.5f} / {1:.5f}".format(lrs_reduced_after_calib[:, i].min(), lrs_reduced_after_calib[:, i].max()))


    print("\nCalculated with original mixtures data:")
    print("---------------------------------------")
    for i, target_class in enumerate(target_classes):
        print("Cllr for {}: {}".format(vec2string(target_class, label_encoder),
                                       round(cllr(lrs_after_calib_mixt[:, i], y_nhot_mixtures, target_class), 4)))
        print("     min/max LR before: {0:.5f} / {1:.5f}".format(lrs_before_calib_mixt[:, i].min(), lrs_before_calib_mixt[:, i].max()))
        print("     min/max LR after: {0:.5f} / {1:.5f}".format(lrs_after_calib_mixt[:, i].min(), lrs_after_calib_mixt[:, i].max()))

    if show:
        plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, title2='y test augmented', show=show)
        plot_histogram_log_lr(lrs_reduced_before_calib, y_test_nhot_augmented_reduced, target_classes, label_encoder, title2='y test augmented reduced', show=show)
        plot_histogram_log_lr(lrs_before_calib_mixt, y_nhot_mixtures, target_classes, label_encoder, title2='y mixtures', show=show)

        if not settings.model == 'MLR':
            plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder, title='after', title2='y test augmented', show=show)
            plot_histogram_log_lr(lrs_reduced_after_calib, y_test_nhot_augmented_reduced, target_classes, label_encoder, title='after', title2='y test augmented reduced', show=show)
            plot_histogram_log_lr(lrs_after_calib_mixt, y_nhot_mixtures, target_classes, label_encoder, title='after', title2='y mixtures', show=show)

            makeplot_hist_density(model.predict_lrs(X_calib_augmented, target_classes, with_calibration=False), y_calib_nhot_augmented, model._calibrators_per_target_class, target_classes, label_encoder, show=show)


def use_repeated_measurements_as_single(X_single, y_nhot_single, y_single):

    N = X_single.size
    X_single_nrp = []
    y_nhot_single_nrp = []
    y_single_nrp = []
    for i in range(N):
        n = X_single[i].shape[0]
        y_nhot_single_i = np.tile(y_nhot_single[i, :], (n, 1))
        y_single_nrp.extend(y_single[i].tolist() * n)
        for j in range(n):
            X_single_nrp.append(X_single[i][j])
            y_nhot_single_nrp.append(y_nhot_single_i[j, :])

    X_single_nrp = np.asarray(X_single_nrp)
    y_nhot_single_nrp = np.asarray(y_nhot_single_nrp)
    y_single_nrp = np.asarray(y_single_nrp)

    return X_single_nrp, y_nhot_single_nrp, y_single_nrp


def only_use_same_combinations_as_in_mixtures(X_augmented, y_nhot, y_nhot_mixtures):
    """
    Make sure that the combinations of cell types present in the mixtures dataset is the
    same in the augmented test dataset.
    """

    unique_mixture_combinations = np.unique(y_nhot_mixtures, axis=0)
    indices = np.array([np.argwhere(np.all(y_nhot == unique_mixture_combinations[i, :], axis=1)).ravel() for i in range(unique_mixture_combinations.shape[0])]).flatten()

    X_reduced = X_augmented[indices, :]
    y_nhot_reduced = y_nhot[indices, :]

    return X_reduced, y_nhot_reduced


def generate_lrs(model, mle, X_train, y_train, X_calib, y_calib, X_test, X_test_reduced, X_mixtures, target_classes):
    """
    When softmax the model must be fitted on labels, whereas with sigmoid the model must be fitted on
    an nhot encoded vector representing the labels. Ensure that labels take the correct form, fit the
    model and predict the lrs before and after calibration for both X_test and X_mixtures.
    """

    if settings.lps: # y_train must be list with labels
        try:
            y_train = mle.nhot_to_labels(y_train)
        except: # already are labels
            pass
    else: # y_train must be nhot encoded labels only for target classes
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

    lrs_reduced_before_calib = model.predict_lrs(X_test_reduced, target_classes, with_calibration=False)
    lrs_reduced_after_calib = model.predict_lrs(X_test_reduced, target_classes)

    lrs_before_calib_mixt = model.predict_lrs(X_mixtures, target_classes, with_calibration=False)
    lrs_after_calib_mixt = model.predict_lrs(X_mixtures, target_classes)

    return lrs_before_calib, lrs_after_calib, lrs_reduced_before_calib, lrs_reduced_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt


def get_accuracy(model, mle, y_true, X, target_classes):
    """
    Predicts labels and ensures that both the true and predicted labels are nhot encoded.
    Calculates the accuracy.

    :return: accuracy: the set of labels predicted for a sample must *exactly* match the
        corresponding set of labels in y_true.
    """

    y_pred = model._classifier.predict(X)

    try:
        y_true = mle.labels_to_nhot(y_true)
    except:
        pass

    try:
        y_pred = mle.labels_to_nhot(y_pred)
    except:
        pass

    if y_true.shape[1] != target_classes.shape[0] and y_pred.shape[1] == target_classes.shape[0]:
        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_true = np.array([np.max(np.array(y_true[:, indices[i]]), axis=1) for i in range(len(indices))]).T

    return accuracy_score(y_true, y_pred)


def print_settings(settings):
    print("augment : {}".format(settings.augment))
    print("binarize : {}".format(settings.binarize))
    print("markers : {}".format(settings.markers))
    print("lps : {}".format(settings.lps))
    print("nsamples : {}".format(settings.nsamples))
    print("test_size : {}".format(settings.test_size))
    print("calibration_size : {}".format(settings.calibration_size))
    print("model : {}".format(settings.model))
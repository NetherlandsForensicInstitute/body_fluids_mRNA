"""
Note: Pipeline, currently not in use!
Usage:
    run.py [--augment] [--markers] [--sigmoid] [--samples <s>] [--parameters <?>] [--test <?>]

Options:
    --augment           If provided, use augmented data to train/calibrate the model with
    --binarize          If provided, make data binary
    --markers           If provided, include all markers
    --sigmoid           If provided, estimates probability fo each individual cell type
    --samples <s>       The number of augmented samples per combination
    --model <?>         The model that the analysis is performed with
    --parameters <?>
    --test <?>          The type of data that is tested on
"""

import pickle
import numpy as np

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from rna.analytics import augment_data, classify_single, cllr, combine_samples
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalClassifier
from rna.utils import string2vec, split_data, replace_labels, vec2string, from_nhot_to_labels
from rna.plotting import plot_histogram_log_lr

from lir.plotting import makeplot_hist_density

def perform_analysis():
    model = MarginalClassifier()

    X_augmented, y_nhot_augmented = augment_data(X_single, y_nhot_single, n_celltypes, n_features,
                                                 N_SAMPLES_PER_COMBINATION, label_encoder,
                                                 from_penile=from_penile)

    model.fit_classifier(X_augmented, replace_labels(y_nhot_augmented))
    loglrs = model.predict_lrs(X_augmented, target_classes)

    model.fit_calibration(X_augmented, y_nhot_augmented, target_classes)
    # pickle.dump(model, open('calibrated_model', 'wb'))
    loglrs_calib = model.predict_lrs(X_augmented, target_classes, with_calibration=True)

    plot_histogram_log_lr(loglrs, y_nhot_augmented, target_classes, show=True)
    plot_histogram_log_lr(loglrs_calib, y_nhot_augmented, target_classes, n_bins=10, title='after', show=True)


def perform_analysis_splitting_data(nfolds, show=False):
    classify_single(X_train, y_train)

    for n in range(nfolds):
        print("Fold {}".format(n))

        model = MarginalClassifier()

        X_train_augmented, y_train_nhot_augmented = \
            augment_data(X_train, y_train, n_celltypes, n_features,
                         N_SAMPLES_PER_COMBINATION, label_encoder,
                         binarize=binarize, from_penile=from_penile)

        # pickle.dump(model, open('mlpmodel', 'wb'))

        X_calibration_augmented, y_calibration_nhot_augmented = \
            augment_data(X_calibrate, y_calibrate, n_celltypes, n_features,
                         N_SAMPLES_PER_COMBINATION, label_encoder,
                         binarize=binarize, from_penile=from_penile)

        model.fit_calibration(X_calibration_augmented, y_calibration_nhot_augmented, target_classes)
        from_nhot_to_labels(y_calibration_nhot_augmented)
        # pickle.dump(model, open('calibrated_model', 'wb'))

        X_test_augmented, y_test_nhot_augmented = \
            augment_data(X_test, y_test, n_celltypes, n_features,
                         N_SAMPLES_PER_COMBINATION_TEST, label_encoder,
                         binarize=binarize, from_penile=from_penile)

        print("Accuracy: {}".format(accuracy_score(replace_labels(y_test_nhot_augmented),
                                                   model._classifier.predict(X_test_augmented))))

        lrs_before_calib = model.predict_lrs(X_test_augmented, target_classes)
        lrs_after_calib = model.predict_lrs(X_test_augmented, target_classes, with_calibration=True)

        for i, target_class in enumerate(target_classes):
            print("Cllr for {}: {}".format(vec2string(target_class, label_encoder),
                                           cllr(lrs_after_calib[:, i], y_test_nhot_augmented, target_class)))

        if show:
            plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, show=show)
            plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder,
                                  density=True, title='after', show=show)

            makeplot_hist_density(model.predict_lrs(X_calibration_augmented, target_classes), y_calibration_nhot_augmented,
                                  model._calibrators_per_target_class, target_classes, label_encoder, show=show)



def perform_on_test_data():
    mlpmodel = pickle.load(open('mlpmodel', 'rb'))
    calibrated_model = pickle.load(open('calibrated_model', 'rb'))
    X_test_augmented, y_test_nhot_augmented = \
        augment_data(X_test, y_test, n_celltypes, n_features,
                     N_SAMPLES_PER_COMBINATION_TEST, label_encoder, from_penile=from_penile)
    lrs_before_calib = calibrated_model.predict_lrs(X_test_augmented, target_classes)
    lrs_after_calib = calibrated_model.predict_lrs(X_test_augmented, target_classes, with_calibration=True)
    plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, show=True)
    plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder,
                          density=True, title='after', show=True)


def perform_tests():
    model = MarginalClassifier(max_iter=1000, epsilon=1e-010)

    X_train_augmented, y_train_nhot_augmented = \
        augment_data(X_train, y_train, n_celltypes, n_features,
                     N_SAMPLES_PER_COMBINATION, label_encoder,
                     binarize=binarize, from_penile=from_penile)

    # Check loss
    y = from_nhot_to_labels(y_train_nhot_augmented)
    model.fit_classifier(X_train_augmented, y)
    plt.plot(model._classifier.loss_curve_)
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    print("Train accuracy: {}".format(accuracy_score(y, model._classifier.predict(X_train_augmented))))
    y_mixt = from_nhot_to_labels(y_nhot_mixtures)
    print("Mixture accuracy: {}".format(accuracy_score(y_mixt[:, 0], model._classifier.predict(X_mixtures))))



if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types?
    from_penile = False
    retrain = True

    # augment=pass
    binarize=True
    markers=False
    # sigmoid=pass
    # samples=pass
    # model=pass
    # parameters=pass
    # test=pass

    N_SAMPLES_PER_COMBINATION = 4
    N_SAMPLES_PER_COMBINATION_TEST = 2

    X_single, y_nhot_single, n_celltypes, n_features, \
    n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types)

    X_mixtures, y_nhot_mixtures, mixture_label_encoder = \
        read_mixture_data(n_celltypes, label_encoder, binarize=binarize,
                          markers=markers)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa',
                          'Vaginal.mucosa and/or Menstrual.secretion']
    target_classes = string2vec(target_classes_str, label_encoder)

    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
        split_data(X_single, y_nhot_single, markers=markers)

    if markers:
        n_features = n_features - 4
        assert sum([X_train[i].shape[1] == n_features for i in range(len(X_train))]) == len(X_train)
        assert sum([X_calibrate[i].shape[1] == n_features for i in range(len(X_calibrate))]) == len(X_calibrate)
        assert sum([X_test[i].shape[1] == n_features for i in range(len(X_test))]) == len(X_test)

    if retrain:
        # perform_analysis()
        # perform_analysis_splitting_data(1)
        perform_tests()
    else:
        perform_on_test_data()






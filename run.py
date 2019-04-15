"""
Run the most important functions
"""
import pickle
import numpy as np

from rna.analytics import augment_data, classify_single
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalClassifier
from rna.utils import string2vec, split_data, replace_labels, vec2string
from rna.plotting import plot_histogram_log_lr

from lir.plotting import makeplot_hist_density


def perform_analysis():
    model = MarginalClassifier()

    X_augmented, y_nhot_augmented = augment_data(X_single, y_nhot_single, n_celltypes, n_features,
                                                 N_SAMPLES_PER_COMBINATION, label_encoder,
                                                 from_penile=from_penile)

    model.fit_classifier(X_augmented, replace_labels(y_nhot_augmented))
    lrs = model.predict_lrs(X_augmented, target_classes)

    model.fit_calibration(X_augmented, y_nhot_augmented, target_classes)
    # pickle.dump(model, open('calibrated_model', 'wb'))
    lrs_calib = model.predict_lrs(X_augmented, target_classes, with_calibration=True)

    plot_histogram_log_lr(lrs, y_nhot_augmented, target_classes, show=True)
    plot_histogram_log_lr(lrs_calib, y_nhot_augmented, target_classes, n_bins=10, title='after', show=True)


def perform_analysis_splitting_data(nfolds, show=False):
    classify_single(X_train, y_train)

    for n in range(nfolds):
        print("Fold {}".format(n))

        model = MarginalClassifier()

        X_train_augmented, y_train_nhot_augmented = \
            augment_data(X_train, y_train, n_celltypes, n_features,
                         N_SAMPLES_PER_COMBINATION, label_encoder, from_penile=from_penile)

        model.fit_classifier(X_train_augmented, replace_labels(y_train_nhot_augmented))

        pickle.dump(model, open('mlpmodel', 'wb'))

        X_calibration_augmented, y_calibration_nhot_augmented = \
            augment_data(X_calibrate, y_calibrate, n_celltypes, n_features,
                         N_SAMPLES_PER_COMBINATION, label_encoder, from_penile=from_penile)

        model.fit_calibration(X_calibration_augmented, y_calibration_nhot_augmented, target_classes)

        pickle.dump(model, open('calibrated_model', 'wb'))

        if show:
            X_test_augmented, y_test_nhot_augmented = \
                augment_data(X_test, y_test, n_celltypes, n_features,
                             N_SAMPLES_PER_COMBINATION_TEST, label_encoder, from_penile=from_penile)

            lrs_before_calib = model.predict_lrs(X_test_augmented, target_classes)
            lrs_after_calib = model.predict_lrs(X_test_augmented, target_classes, with_calibration=True)

            plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, label_encoder, show=True)
            plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, label_encoder,
                                  density=True, title='after', show=True)

            makeplot_hist_density(model.predict_lrs(X_calibration_augmented, target_classes), y_calibration_nhot_augmented,
                                  model._calibrators_per_target_class, target_classes, label_encoder, show=True)


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


if __name__ == '__main__':
    from_penile = False
    retrain = False

    N_SAMPLES_PER_COMBINATION = 100
    N_SAMPLES_PER_COMBINATION_TEST = 50

    X_single, y_nhot_single, n_celltypes_with_penile, n_features, \
    n_per_celltype, label_encoder, markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types)

    n_celltypes = n_celltypes_with_penile - 1

    if not from_penile:
        # necessary if skin penile not included, because label encoder in
        # alphabetical order. As a result columns in incorrect order once
        # skin penile removed from label encoder.
        # TODO: Find more convenient solution.
        i_skinpenile = int(label_encoder.transform(['Skin.penile']))

        label_encoder.classes_ = np.delete(label_encoder.classes_,
                                           int(np.argwhere(label_encoder.classes_ == 'Skin.penile')[0]))

        include = np.where(y_nhot_single[:, i_skinpenile] == 0)[0]
        exclude = np.where(y_nhot_single[:, i_skinpenile] == 1)[0]
        X_single = X_single[include]
        y_nhot_single = np.delete(y_nhot_single, i_skinpenile, axis=1)
        y_nhot_single = np.delete(y_nhot_single, exclude, axis=0)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa',
                          'Vaginal.mucosa and/or Menstrual.secretion']
    target_classes = string2vec(target_classes_str, label_encoder)

    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
        split_data(X_single, y_nhot_single)

    if retrain:
        # perform_analysis()
        perform_analysis_splitting_data(1)
    else:
        perform_on_test_data()






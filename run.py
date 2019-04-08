"""
Run the most important functions
"""
import pickle

from rna.analytics import *
from rna.constants import single_cell_types
from rna.input_output import *
from rna.lr_system import *
from rna.utils import *
from rna.plotting import *

if __name__ == '__main__':
    from_penile = False

    N_SAMPLES_PER_COMBINATION = 4
    N_SAMPLES_PER_COMBINATION_TEST = 2

    X_single, y_nhot_single, n_celltypes_with_penile, n_features, \
    n_per_celltype, markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types)

    n_celltypes = n_celltypes_with_penile - 1

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa',
                          'Vaginal.mucosa and/or Menstrual.secretion']
    target_classes = string2vec(target_classes_str, string2index)

    X_augmented, y_nhot_augmented = augment_data(X_single, y_nhot_single, n_celltypes, n_features,
                                                 N_SAMPLES_PER_COMBINATION, string2index,
                                                 from_penile=from_penile)


    model = MarginalClassifier()
    model.fit(X_augmented, from_nhot_to_labels(y_nhot_augmented))

    lrs = model.predict_lrs(X_augmented, target_classes)

    model.fit_calibration(X_augmented, y_nhot_augmented, target_classes)
    pickle.dump(model, open('calibrated_model', 'wb'))
    lrs_calib = model.predict_lrs(X_augmented, target_classes, with_calibration=True)

    # Apply model to splitted data
    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
        split_data(X_single, y_nhot_single)

    model = MarginalClassifier()

    X_train_augmented, y_train_nhot_augmented = \
        augment_data(X_train, y_train, n_celltypes, n_features,
                     N_SAMPLES_PER_COMBINATION, string2index, from_penile=from_penile)

    model.fit(X_train_augmented, from_nhot_to_labels(y_train_nhot_augmented))

    X_calibration_augmented, y_calibration_nhot_augmented = \
        augment_data(X_calibrate, y_calibrate, n_celltypes, n_features,
                     N_SAMPLES_PER_COMBINATION, string2index, from_penile=from_penile)

    model.fit_calibration(X_calibration_augmented, y_calibration_nhot_augmented, target_classes)

    X_test_augmented, y_test_nhot_augmented = \
        augment_data(X_test, y_test, n_celltypes, n_features,
                     N_SAMPLES_PER_COMBINATION_TEST, string2index, from_penile=from_penile)

    lrs_before_calib = model.predict_lrs(X_test_augmented, target_classes)
    lrs_after_calib = model.predict_lrs(X_test_augmented, target_classes, with_calibration=True)

    plot_histogram_log_lr(lrs_before_calib, y_test_nhot_augmented, target_classes, show=True)
    plot_histogram_log_lr(lrs_after_calib, y_test_nhot_augmented, target_classes, title='after', show=True)

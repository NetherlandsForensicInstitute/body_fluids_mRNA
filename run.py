"""
Run the most important functions
"""
import pickle

from rna.analytics import *
from rna.constants import single_cell_types
from rna.input_output import *
from rna.lr_system import *
from rna.utils import *

if __name__ == '__main__':
    from_penile = False

    N_SAMPLES_PER_COMBINATION = 4

    X_single, y_nhot_single, n_celltypes_with_penile, n_features, \
    n_per_celltype, string2index, index2string, markers, present_celltypes = \
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




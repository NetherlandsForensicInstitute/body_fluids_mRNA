"""
Run the analysis
"""

from rna.analytics import *
from rna.input_output import *
from rna.utils import *


if __name__ == '__main__':
    developing = False
    include_blank = False
    from_penile = False

    N_SAMPLES_PER_COMBINATION = 4

    X_single, y_nhot_single, n_celltypes_with_penile, n_features, n_per_celltype, string2index, index2string = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)

    n_celltypes = n_celltypes_with_penile-1
    celltypes = np.eye(n_celltypes, dtype=int)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']
    target_classes = string2vec(target_classes_str, celltypes, string2index)

    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = split_data(X_single, from_nhot_to_labels(y_nhot_single))

    X_augmented, y_augmented, y_nhot_augmented = augment_data(X_single, y_nhot_single, n_celltypes, n_features, N_SAMPLES_PER_COMBINATION, string2index, from_penile=from_penile)

    from_nhot_to_labels(y_nhot_augmented)




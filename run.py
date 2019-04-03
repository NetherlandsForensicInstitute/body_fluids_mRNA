"""
Run the most important functions
"""

from rna.input_output import *
from rna.utils import *

if __name__ == '__main__':
    developing = False
    include_blank = False

    X_single, y_single, y_nhot_single, n_celltypes_with_penile, n_features, n_per_celltype, string2index, index2string = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)

    n_celltypes = n_celltypes_with_penile-1
    celltypes = np.eye(n_celltypes, dtype=int)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']
    target_classes = string2vec(target_classes_str, celltypes, string2index)


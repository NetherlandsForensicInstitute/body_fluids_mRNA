"""
General calculations.
"""

import numpy as np

from rna.constants import single_cell_types


def string2vec(list_of_strings, label_encoder):
    """
    Converts a list of strings of length N to an N x n_single_cell_types representation of 0s and 1s

    :param list_of_strings: list of strings. Multiple cell types should be separated by and/or
    :param label_encoder: LabelEncoder mapping strings to indices and vice versa
    :return: n_mixures x n_celltypes matrix
    """

    target_classes = np.zeros((len(list_of_strings), len(single_cell_types)))
    for i, list_item in enumerate(list_of_strings):
        celltypes = list_item.split(' and/or ')
        for celltype in celltypes:
            target_classes[i, int(label_encoder.transform([celltype]))] = 1
    return target_classes


def vec2string(target_class, label_encoder):
    """
    Converts a vector of 0s and 1s into a string being one cell type or combined cell types.

    :param target_class: vector with 0s and 1s
    :param label_encoder: LabelEncoder mapping strings to indices and vice versa
    :return: string
    """

    assert np.argwhere(target_class == 0).size + np.argwhere(target_class == 1).size is len(target_class), \
        'target_class contains value(s) other than 0 or 1.'

    if np.sum(target_class) < 2:
        i_celltype = int(np.argwhere(target_class == 1)[0])
        celltype = label_encoder.classes_[i_celltype]
    else:
        i_celltypes = np.argwhere(target_class == 1)
        celltypes = [label_encoder.classes_[int(i_celltype)] for i_celltype in i_celltypes]
        celltype = ' and/or '.join(celltypes)

    return celltype


class MultiLabelEncoder():

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.nhot_of_combinations = make_nhot_matrix_of_combinations(n_classes)

    def nhot_to_labels(self, y_nhot):
        y = np.array([np.argwhere(np.all(self.nhot_of_combinations == y_nhot[i, :], axis=1)).flatten() for i in range(y_nhot.shape[0])])
        return y.ravel()

    def labels_to_nhot(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            n = y.shape[0]
            y_nhot = np.vstack(self.nhot_of_combinations[y[i], :] for i in range(n))
        return y_nhot

    def transform_single(self, y):
        """
        Transforms the MultiLabelEncoded labels into original labels
        """
        y = y.reshape(-1, 1)
        y_transformed = np.zeros_like(y)
        for label in np.unique(y):
            y_transformed[np.argwhere(np.all(y == label, axis=1)).flatten()] = np.log2(label)

        return y_transformed


    def inv_transform_single(self, y):
        """
        Transforms the original labels into the MultiLabelEncoded labels
        """
        y_transformed = np.zeros_like(y)
        for label in np.unique(y):
            y_transformed[np.argwhere(np.all(y == label, axis=1)).flatten()] = 2 ** label

        return y_transformed


def make_nhot_matrix_of_combinations(N):
    """
    Makes nhot encoded matrix with all possible combinations of existing
    single cell types.

    :param N: int
    :return: 2**N x N nhot encoded matrix
    """

    def int_to_binary(i):
        binary = bin(i)[2:]
        while len(binary) < N:
            binary = '0' + binary
        return np.flip([int(j) for j in binary]).tolist()

    return np.array([int_to_binary(i) for i in range(2**N)])
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


def from_nhot_to_labels(y_nhot):
    """
    Converts nhot encoded matrix into list with labels for unique rows.

    :param y_nhot: nhot encoded matrix
    :return: list of length N_samples
    """

    unique_labels = np.flip(np.unique(y_nhot, axis=0), axis=1)
    if np.array_equal(np.sum(unique_labels, axis=1), np.ones(y_nhot.shape[1])):
        y = np.argmax(y_nhot, axis=1)
    elif unique_labels.shape[0] == 2 ** unique_labels.shape[1]:
        # assumes that the nhot encoded matrix first row consists of zero's
        # and a 1 is added each row starting from the right.
        y = []
        for i in range(unique_labels.shape[0]):
            y += [i] * np.where(np.all(y_nhot == unique_labels[i], axis=1))[0].shape[0]
    elif unique_labels.shape[0] == 7:
        # mixtures
        unique_labels = np.unique(y_nhot, axis=0)
        y = np.zeros((y_nhot.shape[0], 1))
        for i in range(unique_labels.shape[0]):
            index = sum([2 ** int(idx) for idx in np.argwhere(unique_labels[i] == 1)])
            y[np.where(np.all(y_nhot == unique_labels[i], axis=1))[0], :] = index
    else:
        raise ValueError("Cannot convert {} encoded matrix into labels.".format(y_nhot))

    return y


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


def print_settings(settings):
    # print("augment : {}".format(settings.augment))
    print("binarize : {}".format(settings.binarize))
    print("markers : {}".format(settings.markers))
    # print("lps : {}".format(settings.binarize))
    print("nsamples : {}".format(settings.nsamples))
    print("test_size : {}".format(settings.test_size))
    print("calibration_size : {}".format(settings.calibration_size))
    # print("model : {}".format(settings.binarize))


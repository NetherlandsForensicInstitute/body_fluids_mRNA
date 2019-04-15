"""
General calculations.
"""

import random

import numpy as np

from rna.constants import single_cell_types

def split_data(X, y_nhot, size=(0.4, 0.2)):
    """
    Splits the originial dataset in three parts. All parts consist of samples from all
    cell types and there is no overlap between samples within the parts.

    :param X:
    :param y_nhot:
    :param size: tuple containing the relative size of the train and test data
    :return:
    """

    def indices_per_class(y):
        """
        Stores the indices belonging to one class in a list and
        returns a list filled with these lists.
        """
        index_classes = sorted(list(np.unique(y, return_index=True)[1]))[1:]
        index_classes1 = index_classes.copy()
        index_classes2 = index_classes.copy()

        index_classes1.insert(0, 0)
        index_classes2.append(len(y))

        index_classes = zip(index_classes1, index_classes2)

        indices_per_class = [[i for i in range(index_class[0], index_class[1])] for index_class in index_classes]

        # put back in correct order
        _, idx = np.unique(y, return_index=True)
        correct_order = y[np.sort(idx)]
        sorted_indices_per_class = indices_per_class.copy()

        for i, true_index in enumerate(correct_order):
            sorted_indices_per_class[true_index] = indices_per_class[i]

        return sorted_indices_per_class


    def define_random_indices(indices, size):
        """
        Randomly defines indices and assigns them to either train
        or calib. For the test set the same samples are included.
        """
        train_size = size[0]
        test_size = size[1]

        test_index = indices[0:int(len(indices) * test_size)]
        left_over_indices = [i for i in indices if i not in test_index]
        train_index = random.sample(left_over_indices, int(train_size * len(indices)))
        calibration_index = [i for i in indices if i not in test_index and i not in train_index]

        return train_index, calibration_index, test_index

    assert sum(size) <= 1.0, 'The sum of the size for the train and test ' \
                            'data must be must be equal to or below 1.0.'

    indices_classes = indices_per_class(from_nhot_to_labels(y_nhot))

    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = ([] for _ in range(6))

    for indices_class in indices_classes:
        indices = [i for i in range(len(indices_class))]
        train_index, calibration_index, test_index = define_random_indices(indices, size)

        X_for_class = X[indices_class]
        y_nhot_for_class = y_nhot[indices_class]

        X_train.extend(X_for_class[train_index])
        y_train.extend(y_nhot_for_class[train_index])
        X_calibrate.extend(X_for_class[calibration_index])
        y_calibrate.extend(y_nhot_for_class[calibration_index])
        X_test.extend(X_for_class[test_index])
        y_test.extend(y_nhot_for_class[test_index])

    print("The actual distribution (train, calibration, test) is ({}, {}, {})".format(
        round(len(X_train)/X.shape[0], 3),
        round(len(X_calibrate)/X.shape[0], 3),
        round(len(X_test)/X.shape[0], 3))
    )

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_calibrate = np.array(X_calibrate)
    y_calibrate = np.array(y_calibrate)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_calibrate, y_calibrate, X_test, y_test


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
    :return: list of length y_nhot.shape[0]
    """

    unique_labels = np.flip(np.unique(y_nhot, axis=0), axis=1)
    if np.array_equal(np.sum(unique_labels, axis=1), np.ones(y_nhot.shape[1])):
        y = np.argmax(y_nhot, axis=1)
    else:
        # assumes that the nhot encoded matrix first row consists of zero's
        # and a 1 is added each row starting from the right.
        y = []
        for i in range(unique_labels.shape[0]):
            y += [i] * np.where(np.all(y_nhot == unique_labels[i], axis=1))[0].shape[0]

    return y

# TODO: Make this function faster
def replace_labels(y_nhot):
    """
    Replace current labels with labels such that in correct order.

    :param y_nhot: nhot encoded matrix
    :return: list of length y_nhot.shape[0]
    """

    switched_labels = from_nhot_to_labels(y_nhot)

    unique_classes = np.unique(y_nhot, axis=0)
    unique_labels = from_nhot_to_labels(unique_classes)

    for j in range(y_nhot.shape[0]):
        for i in range(len(unique_labels)):
            if np.array_equal(y_nhot[j, :], unique_classes[i]):
                switched_labels[j] = unique_labels[i]

    return switched_labels


def vec2string(target_class, label_encoder):
    """
    Converts a vector of 0s and 1s into a string being one cell type or combined cell types.

    :param target_class: vector with 0s and 1s
    :param label_encoder: LabelEncoder mapping strings to indices and vice versa
    :return: string
    """

    assert np.argwhere(target_class == 0).size + np.argwhere(target_class == 1).size is len(target_class), \
        'target_class contains value other than 0 or 1.'

    if np.sum(target_class) < 2:
        i_celltype = int(np.argwhere(target_class == 1)[0])
        celltype = label_encoder.classes_[i_celltype]
    else:
        i_celltypes = np.argwhere(target_class == 1)
        celltypes = [label_encoder.classes_[int(i_celltype)] for i_celltype in i_celltypes]
        celltype = ' and/or '.join(celltypes)

    return celltype







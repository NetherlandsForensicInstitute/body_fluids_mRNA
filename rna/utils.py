"""
General calculations.
"""

import random

import numpy
import numpy as np

from rna.constants import single_cell_types


def create_information_on_classes_to_evaluate(mixture_classes_in_single_cell_type,
                                              classes_map,
                                              class_combinations_to_evaluate,
                                              y_mixtures,
                                              y_mixtures_matrix):
    """
    Generates data structures pertaining to all classes to evaluate, which are
    single cell types and certain combinations thereof

    :param mixture_classes_in_single_cell_type:
    :param classes_map:
    :param class_combinations_to_evaluate:
    :param y_mixtures:
    :param y_mixtures_matrix:
    :return:
    """
    # select relevant classes to evaluate
    mixture_classes_in_classes_to_evaluate = mixture_classes_in_single_cell_type.copy()
    for celltype in list(mixture_classes_in_classes_to_evaluate):
        if celltype not in classes_map:
            del mixture_classes_in_classes_to_evaluate[celltype]

    # change the classes map integers
    classes_map_to_evaluate = classes_map.copy()
    for idx, celltype in enumerate(sorted(classes_map_to_evaluate)):
        classes_map_to_evaluate[celltype] = idx

    y_combi = np.zeros((len(y_mixtures), len(class_combinations_to_evaluate)))
    for i_combination, combination in enumerate(class_combinations_to_evaluate):
        labels = []
        str_combination = ''
        for k, cell_type in enumerate(combination):
            labels += mixture_classes_in_single_cell_type[cell_type]
            if k < len(combination)-1:
                str_combination += cell_type + ' and/or '
            else:
                str_combination += cell_type
        mixture_classes_in_classes_to_evaluate[str_combination] = (list(set(labels)))
        classes_map_to_evaluate[str_combination] = len(classes_map_to_evaluate) + i_combination
        for i in set(labels):
            y_combi[np.where(np.array(y_mixtures) == i), i_combination] = 1

    return mixture_classes_in_classes_to_evaluate, classes_map_to_evaluate, \
           np.append(y_mixtures_matrix, y_combi, axis=1)


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
        Stores the indices beloging to one class in a list and
        returns a list filled with these lists.
        """
        index_classes = list(np.unique(y, return_index=True)[1])[1:]
        index_classes1 = index_classes.copy()
        index_classes2 = index_classes.copy()

        index_classes1.insert(0, 0)
        index_classes2.append(len(y))

        index_classes = zip(index_classes1, index_classes2)

        return [[i for i in range(index_class[0], index_class[1])] for index_class in index_classes]


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

# TODO: Change h0_h1 to h1_h2
# TODO: Check if needed?
def probs_to_lrs(h0_h1_probs, classes_map, log=False):
    """
    Converts probabilities to (log) likelihood ratios.

    :param h0_h1_probs: dictionary with for each cell type two lists with probabilities for h0 and h1
    :param classes_map: dictionary with for each cell type the accompanied indexnumber
    :param log: boolean if True the 10logLRs are calculated
    :return: dictionary with for each cell type two lists with (10log)LRs for h0 and h1
    """
    h0_h1_lrs = {}
    for celltype in sorted(classes_map):
        if log:
            h0_h1_celltype = h0_h1_probs[celltype]
            h0_h1_lrs[celltype] = [np.log10(h0_h1_celltype[i] / (1 - h0_h1_celltype[i])) for i in
                                        range(len(h0_h1_celltype))]

        else:
            h0_h1_celltype = h0_h1_probs[celltype]
            h0_h1_lrs[celltype] = [h0_h1_celltype[i] / (1 - h0_h1_celltype[i]) for i in
                                        range(len(h0_h1_celltype))]

    return h0_h1_lrs

# TODO: Change h0_h1 to h1_h2
# TODO: Check if needed?
def average_per_celltype(h0_h1):
    """
    Calculates the average for all values per cell type per class within celltype.

    :param h0_h1: list filled with dictionaries with for each cell type two lists with values for h0 and h1
    :return: dictionary with for each cell type two lists with the average value for h0 and h1
    """

    celltypes = list(h0_h1[0].keys())

    combined0 = {celltype : [] for celltype in celltypes}
    combined1 = {celltype : [] for celltype in celltypes}

    for values in h0_h1:
        celltypes_test = list(values.keys())
        assert len(celltypes) == len(celltypes_test), 'Number of celltypes compared is different'
        assert False in [celltype in celltypes for celltype in celltypes_test], 'Different celltypes are compared'

        for celltype in celltypes:
            combined0[celltype].append(values[celltype][0].reshape(-1, 1))
            combined1[celltype].append(values[celltype][1].reshape(-1, 1))

    h0_h1_avg_lrs = {}
    for i in range(len(combined0)):
        h0_h1_avg_lrs[celltypes[i]] = (np.mean(combined0[celltypes[i]], axis=0),
                                       np.mean(combined1[celltypes[i]], axis=0))

    return h0_h1_avg_lrs

# TODO: Check if needed?
def sort_calibrators(all_calibrators):

    celltypes = list(all_calibrators[0].keys())
    sorted_calibrators = {celltype : [] for celltype in celltypes}
    for calibrators in all_calibrators:
        for celltype in celltypes:
            sorted_calibrators[celltype].append(calibrators[celltype])

    return sorted_calibrators


def string2vec(list_of_strings, string2index):
    """
    converts a list of strings of length N to an N x n_single_cell_types representation of 0s and 1s
    :param list_of_strings: list of strings. Multiple cell types should be separated by and/or
    :param string2index: dict that converts single cell type string label to index
    :return:
    """
    target_classes = np.zeros((len(list_of_strings), len(single_cell_types)))
    for i, list_item in enumerate(list_of_strings):
        cell_types = list_item.split(' and/or ')
        for cell_type in cell_types:
            target_classes[i, string2index[cell_type]] = 1
    return target_classes


def from_nhot_to_labels(y_nhot):

    unique_labels = np.flip(np.unique(y_nhot, axis=0), axis=1)
    if np.array_equal(np.sum(unique_labels, axis=1), np.ones(y_nhot.shape[1])):
        y = np.argmax(y_nhot, axis=1)
    else:
        y = []
        for i in range(unique_labels.shape[0]):
            y += [i] * np.where(np.all(y_nhot == unique_labels[i], axis=1))[0].shape[0]

    return y
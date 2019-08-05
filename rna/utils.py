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


def remove_markers(X):
    """
    Removes the gender and control markers.
    """
    try:
        X = X[:, :-4]
    except IndexError:
        X = np.array([X[i][:, :-4] for i in range(X.shape[0])])

    return X


def bool2str_binarize(binarize):
    """
    Converts a boolean to a string.

    :param binarize: boolean if True means that the data has been transformed into binary data,
                        if False the data has been normalized.
    :return: str
    """
    if binarize == True:
        return 'bin'
    elif binarize == False:
        return 'norm'


def bool2str_softmax(softmax):
    """
    Converts a boolean to a string.

    :param softmax: boolean if True means that probabilities have been calculated with the softmax function,
                        if False probabilities have been calculated with the sigmoid function.
    :return: str
    """
    if softmax == True:
        return 'soft'
    elif softmax == False:
        return 'sig'


def prior2string(prior, label_encoder):
    """
    Converts a string vector of integers into a string. For example if the vector is '[1 10 1 1 1 1]' this
    will be transformed into 'Cell type 1 10x more likely'.

    :param prior: str of vector representing the distribution
    :param label_encoder:
    :return: str
    """

    # convert string into list of integers
    prior = prior.strip('][').split(', ')
    prior = [int(prior[i]) for i in range(len(prior))]

    if len(np.unique(prior)) == 1:
        return 'Uniform'

    else:
        counts = {prior.count(value): value for value in list(set(prior))}
        value_relevant_prior = counts[1]
        index_of_relevant_prior = prior.index(value_relevant_prior)
        counts.pop(1)
        value_other_priors = list(counts.values())[0]

        if value_relevant_prior > value_other_priors:
            difference = 'more'
            value = value_relevant_prior
        elif value_relevant_prior < value_other_priors:
            difference = 'less'
            value = value_other_priors

        name = label_encoder.inverse_transform([index_of_relevant_prior])[0]

        return '{} {}x {} likely'.format(name, value, difference)
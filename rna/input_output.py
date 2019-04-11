"""
Reads and manipulates datasets.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from rna.constants import string2index


# TODO: include else option when '_rv' not in filename
# TODO: imports file that contains 4 rv per sample without rv's connected to
def read_df(filename, binarize, number_of_replicates=1):
    """
    Reads in an xls file as a dataframe, replacing NA and binarizing if required.
    Returns the original dataframe and a separate list of indices that are the
    replicate numbers that belong to each sample.

    Note that this function only works if '_rv' is in the filename.

    :param filename: name of file to read in
    :param binarize: whether to binarize - use a cutoff value to convert to 0/1
    :return: (dataframe, list of indices)
    """
    if '_rv' in filename:
        # then sure that it includes 'replicate_values'
        df_rv = pd.read_excel(filename, delimiter=';', index_col=0)
        df = df_rv.loc[:, (df_rv.columns.values[:-1])]
        rv = df_rv[['replicate_value']]
        df.fillna(0, inplace=True)
        if binarize:
            df = 1 * (df > 150)
        return df, rv

    else:
        # TODO: What type of data is to be expected?
        # then 'replicate_values' not included, so
        # assume that all samples have 4 replicates
        df = pd.read_excel(filename, delimiter=';', index_col=0)
        df.fillna(0, inplace=True)
        if binarize:
            df = 1 * (df > 150)

        try:
            rv = np.zeros((df.shape[0], 1))
            replicates = [i for i in range(number_of_replicates)] * int(df.shape[0]/number_of_replicates)
            rv[:, 0] = replicates
        except ValueError:
            rv[:, 0] = [i for i in range(number_of_replicates)] * int(df.shape[0] / number_of_replicates) + \
                       [i for i in range(number_of_replicates)][0:df.shape[0] - len(replicates)]
        return df, rv


def get_data_per_cell_type(filename='Datasets/Dataset_NFI_rv.xlsx', single_cell_types=None,
                           ground_truth_known=True, binarize=True, number_of_replicates=None):
    """
    Returns data per specified cell types.

    Note that the samples are saved in a separate numpy array
    based on their replicate indices. As the size differs per
    sample, the array is filled up with zeros to get the correct
    dimension.

    :param filename: name of file to read in, must include "_rv"
    :param single_cell_types: iterable of strings of all single cell types that exist
    :param ground_truth_known: does this data file have labels for the real classes?
    :param binarize: whether to binarize raw measurement values
    :return: (N_single_cell_experimental_samples x N_measurements per sample x
        N_markers array of measurements,
                N_single_cell_experimental_samples array of int labels of which
                    cell type was measured,
                N_cell types,
                N_markers (=N_features),
                dict: cell type name -> cell type index,
                dict: cell type index -> cell type name,
                dict: cell type index -> N_measurements for cell type

    """

    df, rv = read_df(filename, binarize, number_of_replicates)

    if single_cell_types:
        single_cell_types = set(single_cell_types)
    else:
        if not ground_truth_known:
            raise ValueError('if no cell types are provided, ground truth should be known')
        # if not provided, learn the cell types from the data
        all_celltypes = np.array(df.index)
        for celltype in all_celltypes:
            if celltype not in single_cell_types and celltype!='Skin.penile':
                raise ValueError('unknown cell type: {}'.format(celltype))

    n_celltypes_with_penile = len(single_cell_types) + 1
    n_features = len(df.columns)
    n_per_celltype = dict()

    X_single=[]
    if ground_truth_known:
        for celltype in list(single_cell_types) + ['Skin.penile']:
            data_for_this_celltype = np.array(df.loc[celltype])

            if type(rv) == pd.core.frame.DataFrame:
                rvset_for_this_celltype = np.array(rv.loc[celltype]).flatten()
            # TODO: Currently does not work
            elif type(rv) == list:
                rvset_for_this_celltype = rv[rv[:, 1] == celltype, 0]

            n_full_samples, X_for_this_celltype = get_data_for_celltype(celltype, data_for_this_celltype,
                                                                      indices_per_replicate, rvset_for_this_celltype)

            for repeated_measurements in X_for_this_celltype:
                X_single.append(repeated_measurements)
            n_per_celltype[celltype] = n_full_samples

        y_nhot_single = np.zeros((len(X_single), n_celltypes_with_penile))
        end = 0
        for i, celltype in sorted(enumerate(list(single_cell_types) + ['Skin.penile'])):
            i_celltype = string2index[celltype]
            begin = end
            end = end + n_per_celltype[celltype]
            y_nhot_single[begin:end, i_celltype] = 1

    else:
        n_full_samples, X_single = get_data_for_celltype('Unknown', np.array(df), indices_per_replicate, rv)
        y_nhot_single=None

    X_single = np.array(X_single)

    assert X_single.shape[0] == y_nhot_single.shape[0]

    return X_single, y_nhot_single, n_celltypes_with_penile, n_features, n_per_celltype, list(df.columns), list(df.index)


def read_mixture_data(filename, n_celltypes, binarize=True):
    """
    Reads in the experimental mixture data that is used as test data.

    Note that the samples are saved in a separate numpy array
    based on their replicate indices. As the size differs per
    sample, the array is filled up with zeros to get the correct
    dimension.

    :param n_celltypes: int: number of single cell types
        excluding penile skine
    :param binarize: bool: whether to binarize values
    :return: N_samples x N_markers array of measurements NB only one replicate per
                    sample,
                N_samples iterable of mixture class labels - corresponds to the labels
                    used in data augmentation,
                N_samples x N_single_cell_type n_hot encoding of the labels NB in
                    in single cell type space!
                dict: mixture name -> list of int single cell type labels
                dict: mixture class label -> mixture name
    """

    df, rv = read_df(filename, binarize)
    mixture_celltypes = np.array(df.index)

    # initialize
    test_map = defaultdict(list)
    inv_test_map = {}
    n_per_mixture_celltype = dict()

    X_mixtures = []
    y_nhot_mixtures = np.zeros((0, n_celltypes))
    for mixture_celltype in sorted(set(mixture_celltypes)):
        data_for_this_celltype = np.array(df.loc[mixture_celltype], dtype=float)
        rvset_for_this_celltype = np.array(rv.loc[mixture_celltype]).flatten()

        n_full_samples, X_for_this_celltype = get_data_for_celltype(
            mixture_celltype, data_for_this_celltype, indices_per_replicate, rvset_for_this_celltype)

        for repeated_measurements in X_for_this_celltype:
            X_mixtures.append(repeated_measurements)
        n_per_mixture_celltype[mixture_celltype] = n_full_samples

        celltypes = mixture_celltype.split('+')
        class_label = 0
        y_nhot_for_this_celltype = np.zeros(((n_full_samples), n_celltypes))
        for celltype in celltypes:
            test_map[mixture_celltype].append(string2index[celltype])
            class_label += 2 ** string2index[celltype]
            y_nhot_for_this_celltype[:, string2index[celltype]] = 1
        inv_test_map[class_label] = mixture_celltype

        y_nhot_mixtures = np.vstack((y_nhot_mixtures, y_nhot_for_this_celltype))

    X_mixtures = np.array(X_mixtures)

    assert X_mixtures.shape[0] == y_nhot_mixtures.shape[0]

    return X_mixtures, y_nhot_mixtures, test_map, inv_test_map


def indices_per_replicate(end_replicate, last_index):
    """
    Put all indices from replicates that belong to one sample
    in a list and returns one list filled with these lists.
    """
    end_replicate1 = end_replicate.copy()
    end_replicate2 = end_replicate.copy()

    end_replicate1.insert(0, 0)
    end_replicate2.append(last_index)
    all_ends = zip(end_replicate1, end_replicate2)
    indices_per_replicate_set = [[i for i in range(the_end[0], the_end[1])] for the_end in all_ends]

    return indices_per_replicate_set


def get_data_for_celltype(celltype, data_for_this_celltype, indices_per_replicate, rvset_for_this_celltype):

    end_replicate = [i for i in range(1, len(rvset_for_this_celltype)) if
                     rvset_for_this_celltype[i - 1] > rvset_for_this_celltype[i] or
                     rvset_for_this_celltype[i - 1] == rvset_for_this_celltype[i]]
    indices_per_replicate_set = indices_per_replicate(end_replicate, len(rvset_for_this_celltype))
    n_full_samples = len(indices_per_replicate_set)

    n_discarded = 0
    X_for_this_celltype = []
    for idxs in indices_per_replicate_set:
        candidate_samples = data_for_this_celltype[idxs, :]

        # TODO is make this at least one okay?
        if np.sum(candidate_samples[:, -1]) < 1 or np.sum(candidate_samples[:, -2]) < 1 \
                and 'Blank' not in celltype:
            n_full_samples -= 1
            n_discarded += 1
        else:
            X_for_this_celltype.append(candidate_samples)

    print('{} has {} samples (after discarding {} due to QC on structural markers)'.format(
        celltype,
        n_full_samples,
        n_discarded
    ))

    return n_full_samples, X_for_this_celltype



"""
Reads and manipulates datasets.
"""
# TODO: Rename classes
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# TODO: include else option when '_rv' not in filename
# TODO: imports file that contains 4 rv per sample without rv's connected to
def read_df(filename, binarize, number_of_replicates):
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

    df, rv = read_df(filename, binarize, number_of_replicates)

    if single_cell_types:
        celltypes_set = set(single_cell_types)
        # TODO: celltypes_set.remove('Skin.penile') ?
    else:
        if not ground_truth_known:
            raise ValueError('if no cell types are provided, ground truth should be known')
        # if not provided, learn the cell types from the data
        all_celltypes = np.array(df.index)
        # penile skin should be treated separately
        celltypes_set = set(all_celltypes)
        celltypes_set.remove('Skin.penile')

    string2index = {}
    index2string = {}

    for i, celltype in enumerate(sorted(celltypes_set) + ['Skin.penile']):
        string2index[celltype] = i
        index2string[i] = celltype

    n_celltypes_with_penile = len(string2index)
    n_features = len(df.columns)
    n_per_celltype = dict()

    X_single=[]
    if ground_truth_known:
        # y_single = []

        for celltype in sorted(celltypes_set) + ['Skin.penile']:
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
        for n in range(n_celltypes_with_penile):
            i_celltype = string2index[index2string[n]]
            begin = end
            end = end + n_per_celltype[index2string[i_celltype]]
            y_nhot_single[begin:end, i_celltype] = 1

    else:
        n_full_samples, X_single = get_data_for_celltype('Unknown', np.array(df), indices_per_replicate, rv)
        y_nhot_single=None

    X_single = np.array(X_single)

    return X_single, y_nhot_single, n_celltypes_with_penile, n_features, n_per_celltype, string2index, \
           index2string, list(df.columns), list(df.index)


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


def read_mixture_data(filename, n_single_cell_types_no_penile, classes_map, binarize=True):
    """
    Reads in the experimental mixture data that is used as test data.

    Note that the samples are saved in a separate numpy array
    based on their replicate indices. As the size differs per
    sample, the array is filled up with zeros to get the correct
    dimension.

    :param n_single_cell_types_no_penile: int: number of single cell types
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
        all_ends = [[i for i in range(the_end[0], the_end[1])] for the_end in all_ends]

        return all_ends

    # df, rv = read_df('Datasets/Dataset_mixtures_rv.xlsx', binarize)
    df, rv = read_df(filename, binarize)

    # initialize
    class_labels = np.array(df.index)
    test_map = defaultdict(list)
    X_mixtures = []
    y_mixtures = []
    inv_test_map = {}

    rvs = np.array(rv).flatten()
    N_full_samples = len([i for i in range(1, len(rvs)) if rvs[i-1] > rvs[i] or
                          rvs[i-1] == rvs[i]])+1
    y_mixtures_n_hot = np.zeros(
        (N_full_samples, n_single_cell_types_no_penile),
        dtype=int
    )
    n_total = 0

    for clas in sorted(set(class_labels)):
        cell_types = clas.split('+')
        class_label = 0
        data_for_this_label = np.array(df.loc[clas], dtype=float)
        rv_set_per_class = np.array(rv.loc[clas]).flatten()
        end_replicate = [i for i in range(1, len(rv_set_per_class)) if
                         rv_set_per_class[i-1] > rv_set_per_class[i] or
                         rv_set_per_class[i-1] == rv_set_per_class[i]]
        replicate_indices = indices_per_replicate(end_replicate, len(rv_set_per_class))

        n_full_samples = len(replicate_indices)

        for idxs in replicate_indices:
            X_mixtures.append(data_for_this_label[idxs, :])

        for cell_type in cell_types:
            test_map[clas].append(classes_map[cell_type])
            class_label += 2 ** classes_map[cell_type]
            y_mixtures_n_hot[n_total:n_total + n_full_samples, classes_map[cell_type]] = 1
        inv_test_map[class_label] = clas
        n_total += n_full_samples

        y_mixtures += [class_label] * n_full_samples
    X_mixtures = np.array(X_mixtures)

    return X_mixtures, y_mixtures, y_mixtures_n_hot, test_map, inv_test_map
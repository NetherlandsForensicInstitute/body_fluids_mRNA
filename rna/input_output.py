"""
Reads and manipulates datasets.
"""
import csv
import os

import numpy as np
import pandas as pd

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from rna import constants
from rna.analytics import combine_samples
from rna.utils import remove_markers


def read_df(filename, nreplicates=None):
    """
    Reads in an xls file as a dataframe, replacing NA if required. Returns the dataframe containing the data with the
    signal values and a dataframe with the repeated measurements belonging to a sample.

    :param filename: path to the file
    :param nreplicates: number of repeated measurements
    :return: df: pd.DataFrame and rv: pf.DataFrame
    """
    # os.chdir('/Users/Naomi/Documents/Documenten - MacBook Pro van Naomi/statistical_science/jaar_2/internship/method')

    pd.options.mode.chained_assignment = None # to silence warning
    raw_df = pd.read_excel(filename, delimiter=';', index_col=0)
    try:
        rv = raw_df[['replicate_value']]
        df = raw_df.loc[:, (raw_df.columns.values != 'replicate_value')]
        df.fillna(0, inplace=True)
    except KeyError:
        print("Replicate values have not been found and will be added manually."
              "The number of repeated measurements per sample is {}".format(nreplicates))
        df = raw_df
        df.fillna(0, inplace=True)
        unique_celltypes = pd.Series(df.index).unique()
        n_per_celltype = Counter(df.index)

        rv_list = []
        for celltype in unique_celltypes:
            replicates_for_this_celltype = [i for i in range(1, nreplicates + 1)] * int(
                n_per_celltype[celltype] / nreplicates)
            if (n_per_celltype[celltype]/nreplicates).is_integer():
                rv_list.extend(replicates_for_this_celltype)
            else:
                replicates_for_this_celltype = replicates_for_this_celltype + \
                                               [i for i in range(1, nreplicates+1)][0:n_per_celltype[celltype] - len(replicates_for_this_celltype)]
                rv_list.extend(replicates_for_this_celltype)
        rv = pd.DataFrame(rv_list, index=df.index)

    return df, rv


def get_data_per_cell_type(filename='Datasets/Dataset_NFI_rv.xlsx', single_cell_types=None, nreplicates=None,
                           ground_truth_known=True, remove_structural=True):
    """
    Returns data per specified cell types.

    :param filename: name of file to read in, must include "_rv"
    :param single_cell_types: iterable of strings of all single cell types that exist
    :param nreplicates: number of repeated measurements
    :param ground_truth_known: does this data file have labels for the real classes?
    :param remove_structural: bool: if True remove the housekeeping and gender markers
    :return: X_single: (N_single_cell_experimental_samples x N_measurements per sample x N_markers array of measurements,
        y_nhot_single: N_samples x N_single_cell_type n_hot encoding of the labels NB in single cell type space!
        n_celltypes: N_cell types,
        n_features: N_markers (=N_features),
        n_per_celltype: dict: cell type index -> N_measurements for cell type,
        label_encoder: LabelEncoder: cell type index -> cell type name and cell type name -> cell type index,
        list containing al n_features marker names,
        list containing strings of all n_samples labels
    """

    df, rv = read_df(filename, nreplicates)

    label_encoder = LabelEncoder()
    if single_cell_types:
        single_cell_types = list(set(single_cell_types))
        label_encoder.fit(single_cell_types)
    else:
        if not ground_truth_known:
            raise ValueError('if no cell types are provided, ground truth should be known')
        # if not provided, learn the cell types from the data
        all_celltypes = np.array(df.index)
        for celltype in all_celltypes:
            if celltype not in constants.single_cell_types and celltype!='Skin.penile':
                raise ValueError('unknown cell type: {}'.format(celltype))

        label_encoder.fit(all_celltypes)

    n_celltypes = len(single_cell_types)
    n_features = len(df.columns)
    n_per_celltype = dict()

    X_single=[]
    if ground_truth_known:
        for celltype in list(label_encoder.classes_):
            data_for_this_celltype = np.array(df.loc[celltype])
            rvset_for_this_celltype = np.array(rv.loc[celltype]).flatten()
            assert data_for_this_celltype.shape[0] == rvset_for_this_celltype.shape[0]

            n_full_samples, X_for_this_celltype = get_data_for_celltype(celltype, data_for_this_celltype,
                                                                        indices_per_replicate, rvset_for_this_celltype,
                                                                        True)

            for repeated_measurements in X_for_this_celltype:
                X_single.append(repeated_measurements)
            n_per_celltype[celltype] = n_full_samples

        y_nhot_single = np.zeros((len(X_single), n_celltypes))
        end = 0
        for celltype in list(label_encoder.classes_):
            i_celltype = label_encoder.transform([celltype])
            begin = end
            end = end + n_per_celltype[celltype]
            y_nhot_single[begin:end, i_celltype] = 1

        assert np.array(X_single).shape[0] == y_nhot_single.shape[0]

    else:
        n_full_samples, X_single = get_data_for_celltype('Unknown', np.array(df), indices_per_replicate, rv, True)
        y_nhot_single=None

    X_single = np.array(X_single)

    markers = list(df.columns)
    if remove_structural:
        X_single = remove_markers(X_single)
        n_features = n_features-4
        markers = markers[:-4]

    return X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, \
           label_encoder, markers, list(df.index)


def read_mixture_data(n_celltypes, label_encoder, binarize=True, remove_structural=True):
    """
    Reads in the experimental mixture data that is used as test data.

    :param n_celltypes: int: number of single cell types excluding penile skin
    :param label_encoder: LabelEncoder: cell type index -> cell type name and cell type name -> cell type index
    :param binarize: bool: whether to binarize values
    :param remove_structural: bool: whether to remove the housekeeping and gender markers
    :return: X_mixtures: N_samples x N_markers array of measurements NB only one replicate per sample,
        y_nhot_mixtures : N_samples x N_single_cell_type n_hot encoding of the labels NB in in single cell type space!,
        mixture_label_encoder: LabelEncoder: mixture cell type index -> mixture cell type name and
                                             mixture cell type name -> mixture cell type index,
    """

    df, rv = read_df('Datasets/Dataset_mixtures_rv.xlsx')
    mixture_celltypes = np.array(df.index)
    mixture_label_encoder = LabelEncoder()
    mixture_label_encoder.fit(mixture_celltypes)

    if binarize:
        df = 1 * (df > 150)

    n_per_mixture_celltype = dict()
    X_mixtures = []
    y_nhot_mixtures = np.zeros((0, n_celltypes))
    for mixture_celltype in list(mixture_label_encoder.classes_):
        data_for_this_celltype = np.array(df.loc[mixture_celltype], dtype=float)
        rvset_for_this_celltype = np.array(rv.loc[mixture_celltype]).flatten()

        n_full_samples, X_for_this_celltype = get_data_for_celltype(mixture_celltype, data_for_this_celltype,
                                                                    indices_per_replicate, rvset_for_this_celltype,
                                                                    True)

        for repeated_measurements in X_for_this_celltype:
            X_mixtures.append(repeated_measurements)
        n_per_mixture_celltype[mixture_celltype] = n_full_samples

        celltypes = mixture_celltype.split('+')
        y_nhot_for_this_celltype = np.zeros(((n_full_samples), n_celltypes))
        for celltype in celltypes:
            y_nhot_for_this_celltype[:, label_encoder.transform([celltype])] = 1

        y_nhot_mixtures = np.vstack((y_nhot_mixtures, y_nhot_for_this_celltype))

    X_mixtures = combine_samples(X_mixtures)
    if not binarize:
        X_mixtures = X_mixtures / 1000

    if remove_structural:
        X_mixtures = remove_markers(X_mixtures)

    assert X_mixtures.shape[0] == y_nhot_mixtures.shape[0]

    return X_mixtures, y_nhot_mixtures, mixture_label_encoder


def indices_per_replicate(end_replicate, last_index):
    """
    Put all indices from replicates that belong to one sample in a list and returns one list filled with these lists.
    """
    end_replicate1 = end_replicate.copy()
    end_replicate2 = end_replicate.copy()

    end_replicate1.insert(0, 0)
    end_replicate2.append(last_index)
    all_ends = zip(end_replicate1, end_replicate2)
    indices_per_replicate_set = [[i for i in range(the_end[0], the_end[1])] for the_end in all_ends]

    return indices_per_replicate_set


def get_data_for_celltype(celltype, data_for_this_celltype, indices_per_replicate, rvset_for_this_celltype, discard=True):
    """
    Combines the repeated measurements for all samples of the cell type of interest into one sample

    :param celltype: str: cell type
    :param data_for_this_celltype: array of data belonging to mixture cell type without replicated values combined
    :param indices_per_replicate: function that returns all indices from replicates that belong to one sample in a
                                    list and returns one list filled with these lists.
    :param rvset_for_this_celltype: indices of repeated measurements for cell type of interest
    :param discard: bool: whether samples that are invalid should be removed
    :return: n_full_samples: int: the number of samples for this cell type
        X_for_this_celltype: n_full_samples x n_features of data belonging to this cell type
    """

    end_replicate = [i for i in range(1, len(rvset_for_this_celltype)) if
                     rvset_for_this_celltype[i - 1] > rvset_for_this_celltype[i] or
                     rvset_for_this_celltype[i - 1] == rvset_for_this_celltype[i]]
    indices_per_replicate_set = indices_per_replicate(end_replicate, len(rvset_for_this_celltype))
    n_full_samples = len(indices_per_replicate_set)

    n_discarded = 0
    X_for_this_celltype = []
    for idxs in indices_per_replicate_set:
        candidate_samples = data_for_this_celltype[idxs, :]

        if discard:
            # after consultation: keep if at least 50% of housekeeping markers
            # are detected
            if np.sum(candidate_samples[:, -1]) + \
                    np.sum(candidate_samples[:, -2]) < \
                    candidate_samples.shape[0]/2 \
                    and 'Blank' not in celltype:
                n_full_samples -= 1
                n_discarded += 1
            else:
                X_for_this_celltype.append(candidate_samples)
        else:
            X_for_this_celltype.append(candidate_samples)

    return n_full_samples, X_for_this_celltype


def save_data_table(X_single, celltypes, present_markers,
                    save_path):
    with open(save_path, 'w+') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['type'] + present_markers)
        for i, cell_type in enumerate(set(celltypes)):
            samples = X_single[np.array(celltypes)==cell_type]
            if len(samples)>0:
                if len(samples[0].shape)==1:
                    # mixture dataset, replicates are already combined
                    total = np.sum(samples>0,  axis=0)
                    number = len(samples)
                else:
                    total = np.sum(np.array([np.sum(x>0, axis=0) for x in samples]),
                axis=0)
                    number = np.sum([len(x) for x in samples])
                writer.writerow([f'{cell_type.replace(".", " ")} ('
                                 f'{np.sum(np.array(celltypes)==cell_type)})'] +
                                [f'{t:.3f}' for t in total/number])
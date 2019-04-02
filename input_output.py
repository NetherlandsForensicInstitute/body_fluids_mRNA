"""
Reads and manipulates datasets.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def read_df(filename, binarize):
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
        df_rv = pd.read_excel(filename, delimiter=';')
        df = df_rv.loc[:, (df_rv.columns.values[:-1])]
        rv = df_rv[['replicate_value']]
        df.fillna(0, inplace=True)
        if binarize:
            df = 1 * (df > 150)
        return df, rv

    else:
        raise ValueError('Data file should contain column '
                         'of indices, indicated by "rv" in file name.'
                         'Change te filename by concatenating "_rv" only'
                         'if you are sure that such a column is present in '
                         'the data file.')


def get_data_per_cell_type(filename='Datasets/Dataset_NFI_rv.xlsx',
                           binarize=True, developing=False, include_blank=False):
    """
    Returns data per specified cell types.

    Note that the samples are saved in a separate numpy array
    based on their replicate indices. As the size differs per
    sample, the array is filled up with zeros to get the correct
    dimension.

    :param filename: name of file to read in, must include "_rv"
    :param binarize: whether to binarize raw measurement values
    :param developing: if developing, ignore Skin and Blank category for speed
    :param include_blank: whether to include Blank as a separate cell type
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
        all_ends = [[i for i in range(the_end[0], the_end[1])] for the_end in all_ends]

        return all_ends

    df, rv = read_df(filename, binarize)
    class_labels = np.array(df.index)
    # penile skin should be treated separately
    classes_set = set(class_labels)
    classes_set.remove('Skin.penile')

    # initialize
    classes_map = {}
    inv_classes_map = {}
    i = 0

    for clas in sorted(classes_set) + ['Skin.penile']:
        if include_blank or 'Blank' not in clas:
            if not developing or ('Skin' not in clas and 'Blank' not in clas):
                classes_map[clas] = i
                inv_classes_map[i] = clas
                i += 1
    classes = [classes_map[clas] for clas in class_labels if
               (not developing or ('Skin' not in clas and 'Blank' not in clas))
               and (include_blank or 'Blank' not in clas)]
    n_per_class = Counter(classes) # This is not the final number per class!
                                   # Samples may be discarded later on.
    n_features = len(df.columns)

    X_raw = []
    y = []
    for clas in sorted(classes_set) + ['Skin.penile']:
        if include_blank or 'Blank' not in clas:
            if not developing or ('Skin' not in clas and 'Blank' not in clas):
                # Implement which make pairs of replicates
                data_for_this_label = np.array(df.loc[clas])
                rv_set_per_class = np.array(rv.loc[clas]).flatten()

                end_replicate = [i for i in range(1, len(rv_set_per_class)) if
                                  rv_set_per_class[i-1] > rv_set_per_class[i] or
                                  rv_set_per_class[i-1] == rv_set_per_class[i]]
                replicate_indices = indices_per_replicate(end_replicate, len(rv_set_per_class))

                n_full_samples = len(replicate_indices)
                n_discarded = 0

                for idxs in replicate_indices:
                    candidate_samples = data_for_this_label[idxs, :]

                    # TODO is make this at least one okay?
                    if np.sum(candidate_samples[:, -1]) < 1 or \
                            np.sum(candidate_samples[:, -2]) < 1 \
                            and 'Blank' not in clas:
                        n_full_samples -= 1
                        n_discarded += 1
                    else:
                        X_raw.append(candidate_samples)

                print('{} has {} samples (after discarding {} due to QC on '
                      'structural markers)'.format(
                    clas,
                    n_full_samples,
                    n_discarded
                ))
                n_per_class[classes_map[clas]] = n_full_samples
                y += [classes_map[clas]] * n_full_samples

    X_raw = np.array(X_raw)
    n_single_cell_types = len(classes_map)

    return X_raw, y, n_single_cell_types, n_features, classes_map, \
           inv_classes_map, n_per_class


def read_mixture_data(n_single_cell_types_no_penile, classes_map,
                      binarize=True):
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

    df, rv = read_df('Datasets/Dataset_mixtures_rv.xlsx', binarize)

    # initialize
    class_labels = np.array(df.index)
    test_map = defaultdict(list)
    X_mixtures = []
    y_mixtures = []
    inv_test_map = {}

    # TODO: Check if can do this differently
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
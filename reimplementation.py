import math
import pickle
import csv
import os
import random
from collections import Counter, defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from metrics import get_lr_metrics
from calibrations import *
from lir.calibration import KDECalibrator
from lir.util import Xn_to_Xy, Xy_to_Xn


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
    df, rv = read_df(filename, binarize)
    rv_max = rv['replicate_value'].max()
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
                                   # Samples can be discarded later on.
    n_features = len(df.columns)

    # Set the second dimension to rv_max so know for sure all measurements per
    # sample will fit and get no dimension errors.
    X_raw = np.zeros([0, rv_max, n_features])
    y = []
    for clas in sorted(classes_set) + ['Skin.penile']:
        if include_blank or 'Blank' not in clas:
            if not developing or ('Skin' not in clas and 'Blank' not in clas):
                # Implement which make pairs of replicates
                data_for_this_label = np.array(df.loc[clas])
                rv_set_per_class = np.array(rv.loc[clas]).flatten()
                # Save the index when a 'new sample' starts and save in
                # end_replicate. This is when the next integer is lower or equal
                # than the current integer. (e.g. 4 > 1 or 2 == 2).
                end_replicate = [ i for i in range(1, len(rv_set_per_class)) if
                                  rv_set_per_class[i-1] > rv_set_per_class[i] or
                                  rv_set_per_class[i-1] == rv_set_per_class[i]]
                n_full_samples = len(end_replicate)+1
                #TODO variabel aantal samples toestaan? --> wordt nu opgelost
                # in 'combine_samples'.
                data_for_class = np.zeros((n_full_samples, rv_max, n_features))
                n_discarded = 0

                end_replicate.append(len(rv_set_per_class))
                for i in range(n_full_samples):
                    if i == 0:
                        candidate_samples = data_for_this_label[:end_replicate[i], :]
                        numerator = candidate_samples.shape[0]
                        candidate_samples = np.vstack(
                            [candidate_samples,
                             np.zeros([rv_max - candidate_samples.shape[0],
                                       n_features], dtype='int')])
                    else:
                        candidate_samples = data_for_this_label[
                                            end_replicate[i-1]:end_replicate[i], :
                                            ]
                        numerator = candidate_samples.shape[0]
                        candidate_samples = np.vstack(
                            [candidate_samples,
                             np.zeros([rv_max - candidate_samples.shape[0],
                                       n_features], dtype='int')])
                    # Treshold is set depending on the number of replicates per sample.
                    # TODO is make this at least one okay?
                    if np.sum(candidate_samples[:, -1]) < (3*(numerator/4)) or \
                            np.sum(candidate_samples[:, -2]) < (3*(numerator/4)) \
                            and 'Blank' not in clas:
                        n_full_samples -= 1
                        data_for_class = data_for_class[:n_full_samples, :, :]
                        n_discarded += 1
                    else:
                        data_for_class[i - n_discarded, :, :] = candidate_samples

                print('{} has {} samples (after discarding {} due to QC on '
                      'structural markers)'.format(
                    clas,
                    n_full_samples,
                    n_discarded
                ))
                n_per_class[classes_map[clas]] = n_full_samples
                X_raw = np.append(X_raw, data_for_class, axis=0)
                y += [classes_map[clas]] * n_full_samples

    n_single_cell_types = len(classes_map)

    return X_raw, y, n_single_cell_types, n_features, classes_map, \
           inv_classes_map, n_per_class


def combine_samples(data_for_class, n_features):
    """
    Takes a n_samples x rv_max x n_features matrix and returns the
    n_samples x n_markers matrix. The rows including solely zeros are not
    taken into account when combining the samples.

    Note that the latter assumes that there exist no samples in which none of
    the marker values is on. This makes sense as the marker values ACTB and
    18S-rRNA always should be present for the sample to be relevant and is/may
    be deleted if not present.

    :param data_for_class: a n_samples x : x n_features numpy array
    :return: n_samples x N_markers numpy array
    """
    null = np.zeros([n_features])
    data_for_class_mean = np.zeros([data_for_class.shape[0], n_features])
    for i in range(data_for_class.shape[0]):
        delete_rows = []
        if null in data_for_class[i, :, :]:
            for j in range(data_for_class.shape[1]):
                if np.array_equal(data_for_class[i, j, :], null):
                    # Want to delete row j.
                    delete_rows.append(j)
        if len(delete_rows) > 0:
            data_for_class_mean[i, :] = np.mean(
                data_for_class[i, :-1*len(delete_rows), :], axis=0
            )
        else:
            data_for_class_mean[i, :] = np.mean(
                data_for_class[i, :, :], axis=0
            )

    return data_for_class_mean


def classify_single(X, y, inv_classes_map):
    """
    Very simple analysis of single cell type classification, useful as
    preliminary test.
    """
    # classify single classes
    single_samples = combine_samples(X, n_features)
    print('fitting on {} samples, {} features, {} classes'.format(
        len(y),
        single_samples.shape[1],
        len(set(y)))
    )

    X_train, X_test, y_train, y_test = train_test_split(single_samples, y)
    single_model = MLPClassifier(random_state=0)
    single_model.fit(X_train, y_train)
    y_pred = single_model.predict(X_test)
    print('train accuracy for single classes: {}'.format(
        accuracy_score(y_test, y_pred))
    )

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print(cnf_matrix)
    print(inv_classes_map)


def construct_random_samples(X, y, n, classes_to_include, n_features):
    """
    Returns n generated samples that contain classes classes_to_include.
    A sample is generated by random sampling a sample for each class, and adding
    the shuffled replicates

    :param X: N_single_cell_experimental_samples x N_measurements per sample x
        N_markers array of measurements
    :param y: N_single_cell_experimental_samples array of int labels of which
        cell type was measured
    :param n: int: number of samples to generate
    :param classes_to_include:  iterable of int, cell type indices to include
        in the mixtures
    :param n_features: int N_markers (=N_features),
    :return n x n_features array
    """
    if len(classes_to_include) == 0:
        return np.zeros((n, n_features))
    sampled = np.zeros((len(classes_to_include), n, 6, n_features))
    for j, clas in enumerate(classes_to_include):
        n_in_class = sum(np.array(y) == clas)
        data_for_class = np.array([
            X[i, :, :] for i in range(len(X)) if y[i] == clas
        ])
        try:
            sampled[j, :, :, :] = data_for_class[np.random.randint(n_in_class, size=n), :, :]
            # shuffle them
            for i in range(n):
                sampled[j, i, :, :] = sampled[j, i, np.random.permutation(6), :]
        except:
            raise ValueError("The number classes '{}' present in 'y', namely {}, is "
                             "greater/smaller than the number of samples per combination {}".format(
                clas, n_in_class, n))
    combined = np.max(sampled, axis=0)

    return combine_samples(combined, n_features)


def augment_data(X_singles_raw, y_singles, n_single_cell_types, n_features,
                 N_SAMPLES_PER_COMBINATION, classes_map, from_penile=False):
    """
    Generate data for the power set of single cell types

    :param X_singles_raw: N_single_cell_experimental_samples x N_measurements
        per sample x N_markers array of measurements
    :param y_singles: N_single_cell_experimental_samples array of int labels of
        which cell type was measured
    :param n_single_cell_types: int: number of single cell types
    :param n_features: int: N_markers
    :param from_penile: bool: generate samplew that (T) always or (F) never
        also contain penile skin
    :return: N_experiments x N_markers array,
                N_experiment array of int labels for the powerset (=mixtures)
                    classes,
                N_augmented_data_samples x N_single_cell_classes matrix of 0, 1,
                            indicating for each augmented sample which single cell
                            types it was made up of. Does not contain a column for
                            penile skin,
                list of length N_single_cell_classes of lists, that indicate the
                            mixture labels each single cell type features in
    """
    # Generate more data
    if from_penile == False:
        if 'Skin.penile' in classes_map:
            del classes_map['Skin.penile']

    X = np.zeros((0, n_features))
    y = []
    n_single_cell_types_not_penile = n_single_cell_types - 1
    y_n_hot = np.zeros((
        2 ** n_single_cell_types_not_penile * N_SAMPLES_PER_COMBINATION,
        n_single_cell_types
    ), dtype=int)
    mixtures_containing_single_cell_type = {celltype: [] for celltype in classes_map}
    #mixtures_containing_single_cell_type_before = [[] for _ in range(n_single_cell_types_not_penile)]

    for i in range(2 ** n_single_cell_types_not_penile):
        binary = bin(i)[2:]
        while len(binary) < n_single_cell_types_not_penile:
            binary = '0' + binary
        classes_in_current_mixture = []
        for j, celltype in enumerate(classes_map):
            if binary[-j - 1] == '1':
                classes_in_current_mixture.append(j)
                mixtures_containing_single_cell_type[celltype].append(int(i))
                y_n_hot[i * N_SAMPLES_PER_COMBINATION:(i + 1) * N_SAMPLES_PER_COMBINATION, j] = 1
        if from_penile:
            # also (always) add penile skin samples
            y_n_hot[i * N_SAMPLES_PER_COMBINATION:(i + 1) * N_SAMPLES_PER_COMBINATION, n_single_cell_types - 1] = 1
        X = np.append(X, construct_random_samples(
            X_singles_raw, y_singles, N_SAMPLES_PER_COMBINATION, classes_in_current_mixture, n_features), axis=0)
        y += [i] * N_SAMPLES_PER_COMBINATION

    return X, y, y_n_hot[:, :n_single_cell_types_not_penile], \
           mixtures_containing_single_cell_type


def evaluate_model(model, dataset_label, X, y, y_n_hot, labels_in_class, classes_map, MAX_LR):
    """
    Computes metrics for performance of the model on dataset X, y

    :param model: sklearn-like model to evaluate
    :param dataset_label:
    :param X:
    :param y:
    :param y_n_hot:
    :param labels_in_class:
    :return: iterable with for each class a list of len 2, with scores for all
        h1 and h2 scenarios
    """
    print(X.shape)
    y_pred = model.predict(X)
    print('{} accuracy for mixtures: {}'.format(
        dataset_label, accuracy_score(y, y_pred)))

    y_prob = model.predict_proba(X)
    h1_h2_probs_per_class = {}
    # marginal for each single class sample
    prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, labels_in_class, classes_map, MAX_LR)
    for j in range(y_n_hot.shape[1]):
        # get the probability per single class sample
        total_proba = prob_per_class[:, j]
        if sum(total_proba) > 0:
            probas_without_cell_type = total_proba[y_n_hot[:, j] == 0]
            probas_with_cell_type = total_proba[y_n_hot[:, j] == 1]
            # print(inv_classes_map[j], np.quantile(probas_without_cell_type, [0.05, .25, .5, .75, .95]),
            #       np.quantile(probas_with_cell_type, [0.05, .25, .5, .75, .95]))
            h1_h2_probs_per_class[j] = (probas_with_cell_type, probas_without_cell_type)

    return h1_h2_probs_per_class


def convert_prob_per_mixture_to_marginal_per_class(prob, labels_in_class, classes_map, MAX_LR):
    """
    Converts n_samples x n_mixture_classes matrix of probabilities to a
    n_samples x n_classes_of_interest matrix, by summing over the relevant
    mixtures.

    :param prob: n_samples x n_mixture_classes matrix of probabilities
    :param labels_in_class: iterable of len n_classes_of_interest. For each
        class, the list of mixture classes that contain the class of interest
        are given.
    :return: n_samples x n_classes_of_interest matrix of probabilities
    """
    # TODO: make this function work with labels_in_class as dictionary
    res_prob = np.zeros((prob.shape[0], len(labels_in_class)))
    for j in range(res_prob.shape[1]):
        celltype = list(classes_map.keys())[list(classes_map.values()).index(j)]
        if len(labels_in_class[celltype]) > 0:
            res_prob[:, j] = np.sum(prob[:, labels_in_class[celltype]], axis=1)
    epsilon = 10 ** -MAX_LR
    res_prob = np.where(res_prob > 1 - epsilon, 1 - epsilon, res_prob)
    res_prob = np.where(res_prob < epsilon, epsilon, res_prob)

    return res_prob


def read_mixture_data(n_single_cell_types_no_penile, n_features, classes_map,
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
    # read test data
    df, rv = read_df('Datasets/Dataset_mixtures_rv.xlsx', binarize)
    rv_max = rv['replicate_value'].max()

    # initialize
    class_labels = np.array(df.index)
    test_map = defaultdict(list)
    X_mixtures = np.zeros((0, rv_max, n_features))
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
        n_full_samples = len(end_replicate)+1

        data_for_mixt_class = np.zeros((n_full_samples, rv_max, n_features))
        end_replicate.append(len(rv_set_per_class))
        for i in range(n_full_samples):
            if i == 0:
                sample = data_for_this_label[:end_replicate[i], :]
                data_for_mixt_class[i, :, :] = np.vstack([
                    sample,
                    np.zeros([rv_max - sample.shape[0], n_features],
                    dtype='int')])
            else:
                sample = data_for_this_label[end_replicate[i-1]:end_replicate[i], :]
                data_for_mixt_class[i, :, :] = np.vstack([
                    sample,
                    np.zeros([rv_max - sample.shape[0], n_features],
                    dtype='int')])

        for cell_type in cell_types:
            test_map[clas].append(classes_map[cell_type])
            class_label += 2 ** classes_map[cell_type]
            y_mixtures_n_hot[n_total:n_total + n_full_samples, classes_map[cell_type]] = 1
        inv_test_map[class_label] = clas
        n_total += n_full_samples
        X_mixtures = np.append(X_mixtures, data_for_mixt_class, axis=0)
        y_mixtures += [class_label] * data_for_mixt_class.shape[0]

    return X_mixtures, y_mixtures, y_mixtures_n_hot, test_map, inv_test_map


def boxplot_per_single_class_category(X_augmented_test,
                                      y_augmented_matrix,
                                      classes_to_evaluate,
                                      mixtures_in_classes_of_interest,
                                      class_combinations_to_evaluate):
    """
    for single cell type, plot the distribution of marginal LRs for each cell type,
    as well as for specified combinations of classes.

    :param X_augmented_test: N_samples x N_markers array of observations
    :param y_augmented_matrix: N_samples x (N_single_cell_types + N_combos)
        n_hot encoding
    :param classes_to_evaluate: list of str, names of classes to evaluate
    :param mixtures_in_classes_of_interest: list of lists, specifying for each
        class in classes_to_evaluate which
    mixture labels contain these
    :param class_combinations_to_evaluate: list of lists of int, specifying
        combinations of single cell types to consider
    :return: None
    """
    n_single_classes_to_draw = y_augmented_matrix.shape[1]
    y_prob = model.predict_proba(X_augmented_test)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, mixtures_in_classes_of_interest, classes_map, MAX_LR)
    log_lrs_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
    plt.subplots(2, 5, figsize=(18, 9))
    for i in range(n_single_classes_to_draw):
        indices = [j for j in range(y_augmented_matrix.shape[0]) if
                   y_augmented_matrix[j, i] == 1 and sum(y_augmented_matrix[j, :]) == 1]
        plt.subplot(2, 5, i + 1)
        plt.xlim([-MAX_LR -.5, MAX_LR+.5])
        bplot = plt.boxplot(log_lrs_per_class[indices, :], vert=False,
                            labels=classes_to_evaluate, patch_artist=True)
        colors = ['white'] * (n_single_classes_to_draw + 1)
        colors[i] = 'black'
        for j, comb in enumerate(class_combinations_to_evaluate):
            if inv_classes_map[i] in comb:
                colors[n_single_classes_to_draw + j] = 'black'
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(inv_classes_map[i])
    plt.savefig('singles boxplot')


def plot_for_experimental_mixture_data(X_mixtures,
                                       y_mixtures,
                                       y_mixtures_matrix,
                                       inv_test_map,
                                       classes_to_evaluate,
                                       mixtures_in_classes_of_interest,
                                       n_single_cell_types_no_penile,
                                       dists):
    """
    for each mixture category that we have measurements on, plot the
    distribution of marginal LRs for each cell type, as well as for the special
    combinations (eg vaginal+menstrual) also plot LRs as a function of distance
    to nearest data point also plot experimental measurements together with LRs
    found and distance in a large matrix plot

    :param X_mixtures: N_experimental_mixture_samples x N_markers array of
        observations
    :param y_mixtures: N_experimental_mixture_samples array of int mixture labels
    :param y_mixtures_matrix:  N_experimental_mixture_samples x
        (N_single_cell_types + N_combos) n_hot encoding
    :param inv_test_map: dict: mixture label -> mixture name
    :param classes_to_evaluate: list of str, classes to evaluate
    :param mixtures_in_classes_of_interest:  list of lists, specifying for each
        class in classes_to_evaluate which mixture labels contain these
    :param n_single_cell_types_no_penile: int: number of single cell types
        excluding penile skin
    :param dists: N_experimental_mixture_samples iterable of distances to
        nearest augmented data point. Indication of whether the point may be an
        outlier (eg measurement error or problem with augmentation scheme)
    """
    y_prob = model.predict_proba(X_mixtures)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, mixtures_in_classes_of_interest, classes_map, MAX_LR)

    log_lrs_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
    plt.subplots(3, 3, figsize=(18, 9))
    for i, i_clas in enumerate(set(y_mixtures)):
        indices_experiments = [j for j in range(len(y_mixtures)) if y_mixtures[j] == i_clas]
        plt.subplot(3, 3, i + 1)
        plt.xlim([-MAX_LR - .5, MAX_LR + .5])
        bplot = plt.boxplot(log_lrs_per_class[indices_experiments, :], vert=False,
                            labels=classes_to_evaluate, patch_artist=True)

        for j, (patch, cla) in enumerate(zip(bplot['boxes'], classes_to_evaluate)):
            if j < n_single_cell_types_no_penile:
                # single cell type
                if cla in inv_test_map[i_clas]:
                    patch.set_facecolor('black')
            else:
                # sample 'Vaginal.mucosa and/or Menstrual.secretion'
                for comb_class in cla.split(' and/or '):
                    if comb_class in inv_test_map[i_clas]:
                        patch.set_facecolor('black')
        plt.title(inv_test_map[i_clas])
    plt.savefig('mixtures_boxplot')

    plt.subplots(3, 3, figsize=(18, 9))
    for i in range(y_mixtures_matrix.shape[1]):
        plt.subplot(3, 3, i + 1)
        plt.ylim([-MAX_LR - .5, MAX_LR + .5])
        plt.scatter(
            dists + np.random.random(len(dists)) / 20,
            log_lrs_per_class[:, i],
            color=['red' if iv else 'blue' for iv in y_mixtures_matrix[:, i]],
            alpha=0.1
        )
        plt.ylabel('LR')
        plt.xlabel('distance to nearest data point')
        plt.title(classes_to_evaluate[i])
    plt.savefig('LRs_as_a_function_of_distance')

    plt.figure()
    plt.matshow(
        np.append(
            np.append(X_mixtures, log_lrs_per_class, axis=1),
            np.expand_dims(np.array([d*5 for d in dists]), axis=1),
            axis=1))
    plt.savefig('mixtures binned data and log lrs')


def calculate_scores(X, model, mixtures_in_classes_of_interest, n_features, MAX_LR, log):
    """
    Calculates the (log) likehood ratios.

    :param model: model that has been trained
    :param X: numpy array used to predict the probabilites of labels with
    :param mixtures_in_classes_of_interest:
    :param log: boolean if True log LRs are calculated
    :return: the log likelihood ratios for all samples and makers
    """
    if len(X.shape) > 2:
        X = combine_samples(X, n_features)

    y_prob = model.predict_proba(X)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, mixtures_in_classes_of_interest, classes_map, MAX_LR)

    if log:
        log_scores_per_class = np.log10(y_prob_per_class / (1 - y_prob_per_class))
        return log_scores_per_class

    else:
        scores_per_class = y_prob_per_class / (1 - y_prob_per_class)
        return scores_per_class


def plot_data(X):
    """
    plots the raw data points

    :param X: N_samples x N_observations_per_sample x N_markers measurements
    """
    plt.matshow(combine_samples(X, n_features))
    plt.savefig('single_cell_type_measurements_after_QC')


def plot_calibration(h1_h2_scores, classes_to_evaluate):
    """
    Print metrics on and generate plots on calibration NB the confidence
    intervals appear to still have issues

    :param h1_h2_scores: iterable with for each class to evaluate a list of len
        two, containing scores for h1 and h2
    :param classes_to_evaluate: list of str classes to evaluate
    """
    plt.subplots(2, 5, figsize=(18, 9))
    all_bins0 = defaultdict(int)
    all_bins1 = defaultdict(int)
    for j, (h1_scores, h2_scores) in h1_h2_scores.items():
        plt.subplot(2, 5, j + 1)
        h1_lrs = h1_scores / (1 - h1_scores)
        h2_lrs = h2_scores / (1 - h2_scores)
        if len(h1_scores) > 0:
            m = get_lr_metrics(h1_scores=h1_scores,
                               h2_scores=h2_scores,
                               h1_lrs=h1_lrs,
                               h2_lrs=h2_lrs,
                               hp_prior=0.5
                               )
            print(classes_to_evaluate[j], ['{}: {}'.format(
                a[0], round(a[1], 2)) for a in m])
        scale = 10
        bins0 = defaultdict(float)
        for v in h2_lrs:
            v = max(v, 10**-MAX_LR)
            v = min(v, 10**MAX_LR)
            bins0[int(round(math.log(v, scale)))] += 1.0
        for k, b in bins0.items():
            all_bins0[k] += b
        bins1 = defaultdict(float)
        for v in h1_lrs:
            v = max(v, 10**-MAX_LR)
            v = min(v, 10**MAX_LR)
            bins1[int(round(math.log(v, scale)))] += 1.0
        for k, b in bins1.items():
            all_bins1[k] += b

        std_err0, bins0 = transform_counts(bins0, len(h2_lrs), scale, True)
        std_err1, bins1 = transform_counts(bins1, len(h1_lrs), scale, False)

        bins0_x, bins0_y = zip(*sorted(bins0.items()))
        if len(std_err0) > 0:
            bins_se0_x, vals = zip(*sorted(std_err0.items()))
            bins_se0_y, y0 = zip(*vals)
            plt.errorbar(np.array(bins_se0_x) + .15, bins_se0_y, yerr=y0)
        plt.bar(np.array(bins0_x) - .15, bins0_y, label='h2 (0)', width=.3)
        if len(bins1) > 0:
            bins1_x, bins1_y = zip(*sorted(bins1.items()))
            plt.bar(np.array(bins1_x) + .15, bins1_y, label='h1 (1)', width=.3, color='r')
            if len(std_err1) > 0:
                bins_se1_x, vals = zip(*sorted(std_err1.items()))
                bins_se1_y, y1 = zip(*vals)
                plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y, color='r')
        # plt.legend()
        plt.title(classes_to_evaluate[j])
    plt.savefig('calibration separate')

    plt.figure()
    all_std_err0, all_bins0 = transform_counts(
        all_bins0, sum([len(b[1]) for a, b in h1_h2_scores.items()]),
        scale, True
    )

    all_std_err1, all_bins1 = transform_counts(
        all_bins1, sum([len(b[0]) for a, b in h1_h2_scores.items()]),
        scale, False)

    bins0_x, bins0_y = zip(*sorted(all_bins0.items()))
    plt.bar(np.array(bins0_x) - .15, bins0_y, label='h2 (0)', width=.3)
    if len(all_std_err0) > 0:
        bins_se1_x, vals = zip(*sorted(all_std_err0.items()))
        bins_se1_y, y1 = zip(*vals)
        plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y)
    bins1_x, bins1_y = zip(*sorted(all_bins1.items()))
    plt.bar(np.array(bins1_x) + .15, bins1_y, label='h1 (1)', width=.3, color='r')
    if len(all_std_err1) > 0:
        bins_se1_x, vals = zip(*sorted(all_std_err1.items()))
        bins_se1_y, y1 = zip(*vals)
        plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y, color='r')
    plt.legend()
    plt.title('all')
    plt.savefig('calibration all')


def transform_counts(bins, n_obs, scale, is_h2):
    """
    Transform counts so h1 and h2 fractions can be visually compared
    if the score is 'correct' (ie log > 0 for h1 and < 0 for h2), just take the
    fraction if the score is 'incorrect', multiply by how much more often the
    score should occur in the 'correct' scenario, ie by 10**the value of the
    score also provides an (apparently incorrect) standard error for each bin

    :param bins: dict: rounded score -> count
    :param n_obs: int: total number of observations
    :param scale: logscale, eg 10
    :param is_h2: whether these are the counts for h2 (ie False -> h1)
    :return: dict: rounded score -> (standard error * adjustment factor,
                    adjustment factor),
                dict: rounded score -> (possibly adjusted) fraction
    """
    std_err = {}
    for x in bins:
        if (is_h2 and x <= 0) or (not is_h2 and x >= 0):
            bins[x] *= 1 / n_obs
        else:
            p = bins[x] / n_obs
            adjusted_y = bins[x] / n_obs * (scale ** abs(x))
            std_err[x] = (math.sqrt(p * (1 - p) * n_obs) * adjusted_y, adjusted_y)
            bins[x] = adjusted_y

    return std_err, bins


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
    mixture_classes_in_classes_to_evaluate = mixture_classes_in_single_cell_type
    y_combi = np.zeros((len(y_mixtures), len(class_combinations_to_evaluate)))
    for i_combination, combination in enumerate(class_combinations_to_evaluate):
        labels = []
        for cell_type in combination:
            # TODO: make this work with mixture_classes_in_single_cell_type as dictionary
            labels += mixture_classes_in_single_cell_type[classes_map[cell_type]]
        mixture_classes_in_classes_to_evaluate.append(list(set(labels)))
        for i in set(labels):
            y_combi[np.where(np.array(y_mixtures) == i), i_combination] = 1

    return mixture_classes_in_classes_to_evaluate, \
           np.append(y_mixtures_matrix, y_combi, axis=1)


def getNumeric(prompt):
    while True:
        response = input(prompt)
        try:
            return int(response)
        except ValueError:
            print("Please enter a number.")


def manually_refactor_indices():
    """
    Takes the single cell type dataset with two sheets of which one is metadata.
    From the information the replicate numbers are retrieved and an extra column
    is added to the dataframe. If it is not clear to which replicate  sample
    belongs, in a python shell an integer can be set manually.

    :return: excel file with a column "replicate_values" added
    """

    xls = pd.ExcelFile('Datasets/Dataset_NFI_adj.xlsx')
    sheet_to_df_map = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    relevant_column = list(zip(*list(sheet_to_df_map['Samples + details'].index)))[3]
    shortened_names = [relevant_column[i][-3:] for i in range(len(relevant_column))]
    replicate_values = OrderedDict([])
    indexes_to_be_checked = []
    for i in range(len(shortened_names)):
        try:
            replicate_values[i] = int(shortened_names[i][-1])
        except ValueError:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]
        if shortened_names[i] in ['0.3', '.75', '0.5', 'a_1', 'n_1', 'n_4']:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]
        if shortened_names[i][-1] in ['0', '6', '7']:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]

    # Iterate over the index and manually adjust
    replace_replicate = []
    for i in range(len(indexes_to_be_checked)):
        print("The middle rowname of the following rownames should get assigned a replicate number:",
              relevant_column[indexes_to_be_checked[i] - 1:indexes_to_be_checked[i] + 2])
        replace_replicate.append(getNumeric("Give the replicate number"))
    for i in range(len(indexes_to_be_checked)):
        replicate_values[indexes_to_be_checked[i]] = replace_replicate[i]
    replicates = list(replicate_values.values())
    replicates.insert(0, "replicate_value")
    csvfile = "Datasets/replicate_numbers_single.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in replicates:
            writer.writerow([val])

    sheet_to_df_map['Samples + details'].to_csv('Datasets/Dataset_NFI_meta.csv', encoding='utf-8')
    sheet_to_df_map['Data uitgekleed'].to_csv('Datasets/Dataset_NFI_adj.csv', encoding='utf-8')
    a = pd.read_csv("Datasets/Dataset_NFI_meta.csv", delimiter=',')
    b = pd.read_csv("Datasets/Dataset_NFI_adj.csv", delimiter=',')
    c = pd.read_csv("Datasets/replicate_numbers_single.csv")
    mergedac = pd.concat([a, c], axis=1)
    for i in range(len(mergedac.columns.values)):
        if 'Unnamed' in mergedac.columns.values[i]:
            mergedac.columns.values[i] = None
    mergedbc = pd.concat([b, c], axis=1)
    for i in range(len(mergedbc.columns.values)):
        if 'Unnamed' in mergedbc.columns.values[i]:
            mergedbc.columns.values[i] = None
    mergedac.to_excel("Datasets/Dataset_NFI_meta_rv.xlsx", index=False)
    mergedbc.to_excel("Datasets/Dataset_NFI_rv.xlsx", index=False)
    os.remove("Datasets/Dataset_NFI_meta.csv")
    os.remove("Datasets/Dataset_NFI_adj.csv")
    os.remove("Datasets/replicate_numbers_single.csv")


def manually_refactor_indices_mixtures():
    """
    Takes the mixture cell type dataset with two sheets of which one is metadata.
    From the information the replicate numbers are retrieved and an extra column
    is added to the dataframe. If it is not clear to which replicate  sample
    belongs, in a python shell an integer can be set manually.

    :return: excel file with a column "replicate_values" added
    """
    xls = pd.ExcelFile('Datasets/Dataset_mixtures_adj.xlsx')
    sheet_to_df_map = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    relevant_column = list(zip(*list(sheet_to_df_map['Mix + details'].index)))[3]
    shortened_names = [relevant_column[i][-3:] for i in range(len(relevant_column))]
    replicate_values = OrderedDict([])
    indexes_to_be_checked = []

    for i in range(len(shortened_names)):
        try:
            replicate_values[i] = int(shortened_names[i][-1])
        except ValueError:
            print("Some samples do not have clear replicate numbers")
    replicates = list(replicate_values.values())
    replicates.insert(0, "replicate_value")
    csvfile = "Datasets/replicate_numbers_mixture.csv"

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in replicates:
            writer.writerow([val])

    sheet_to_df_map['Mix + details'].to_csv('Datasets/Dataset_mixture_meta.csv', encoding='utf-8')
    sheet_to_df_map['Mix data uitgekleed'].to_csv('Datasets/Dataset_mixture_adj.csv', encoding='utf-8')
    a = pd.read_csv("Datasets/Dataset_mixture_meta.csv", delimiter=',')
    b = pd.read_csv("Datasets/Dataset_mixture_adj.csv", delimiter=',')
    c = pd.read_csv("Datasets/replicate_numbers_mixture.csv")
    mergedac = pd.concat([a, c], axis=1)
    for i in range(len(mergedac.columns.values)):
        if 'Unnamed' in mergedac.columns.values[i]:
            mergedac.columns.values[i] = None
    mergedbc = pd.concat([b, c], axis=1)
    for i in range(len(mergedbc.columns.values)):
        if 'Unnamed' in mergedbc.columns.values[i]:
            mergedbc.columns.values[i] = None
    mergedac.to_excel("Datasets/Dataset_mixtures_meta_rv.xlsx", index=False)
    mergedbc.to_excel("Datasets/Dataset_mixtures_rv.xlsx", index=False)
    os.remove("Datasets/Dataset_mixture_meta.csv")
    os.remove("Datasets/Dataset_mixture_adj.csv")
    os.remove("Datasets/replicate_numbers_mixture.csv")


def split_data(X, y):
    '''
    Splits the data more or less in half per cell type and assigns to either one of
    two datasets.

    :param X: numpy array containing single cell type samples
    :param y: numpy array containing labels of cell types
    :return: two datasets containg splitted X into train and calibration set and same
        holds for y.
    '''
    index_classes = list(np.unique(y, return_index=True)[1])[1:]
    index_classes.append(len(y))

    # initialize
    X_raw_singles_train = np.zeros([0, X.shape[1], X.shape[2]])
    y_raw_singles_train = []
    X_raw_singles_calibrate = np.zeros([0, X.shape[1], X.shape[2]])
    y_raw_singles_calibrate = []

    for i, idx in enumerate(index_classes):
        if i == 0:
            mylist = list(np.linspace(0, idx - 1, idx, dtype=int))
            # pick random half of the samples per class and use for train sample
            train_index = random.sample(mylist, int(idx / 2))
            # use other half for calibration data
            calibration_index = [x for x in mylist if x not in train_index]

            X_raw_singles_train = np.append(X_raw_singles_train, X[:idx][train_index], axis=0)
            y_raw_singles_train.extend([y[:idx][k] for k in train_index])
            X_raw_singles_calibrate = np.append(X_raw_singles_calibrate, X[:idx][calibration_index], axis=0)
            y_raw_singles_calibrate.extend([y[:idx][k] for k in calibration_index])
        else:
            j = index_classes[i] - index_classes[i - 1]
            mylist = list(np.linspace(0, j - 1, j, dtype=int))
            # pick random half of the samples per class and use for train sample
            train_index = random.sample(mylist, int(j / 2))
            # use other half for calibration data
            calibration_index = [x for x in mylist if x not in train_index]

            X_raw_singles_train = np.append(
                X_raw_singles_train, X[index_classes[i-1]:index_classes[i]][train_index], axis=0)
            y_raw_singles_train.extend([y[index_classes[i-1]:index_classes[i]][k] for k in train_index])
            X_raw_singles_calibrate = np.append(
                X_raw_singles_calibrate, X[index_classes[i-1]:index_classes[i]][calibration_index], axis=0)
            y_raw_singles_calibrate.extend([y[index_classes[i-1]:index_classes[i]][k] for k in calibration_index])

    return X_raw_singles_train, y_raw_singles_train, X_raw_singles_calibrate, y_raw_singles_calibrate


if __name__ == '__main__':
    developing = False
    include_blank = False
    unknown_replicatenumbers_single = False
    unknown_replicatenumbers_mixture = False

    # Assign the correct replicates to the same sample for the single cell types.
    if unknown_replicatenumbers_single:
        manually_refactor_indices()

    # Assign the correct replicates to the same sample for the mixture cell types.
    if unknown_replicatenumbers_mixture:
        manually_refactor_indices_mixtures()

    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map, inv_classes_map, n_per_class = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)
    plot_data(X_raw_singles)
    n_folds = 2
    N_SAMPLES_PER_COMBINATION = 50
    MAX_LR=10
    from_penile = False
    retrain = True
    type_train_data = 'calibration' #or 'train'
    model_file_name = 'mlpmodel'
    if from_penile:
        model_file_name+='_penile'

    # which classes should we compute marginals for? all single cell types and a 'contains vaginal' class?
    # '-1' to avoid the penile skin
    single_cell_classes = [inv_classes_map[j] for j in range(n_single_cell_types - 1)]
    class_combinations_to_evaluate = [['Vaginal.mucosa', 'Menstrual.secretion']]
    classes_to_evaluate = single_cell_classes + [' and/or '.join(comb) for comb in class_combinations_to_evaluate]

    # Split the data in two equal parts: for training and calibration
    X_raw_singles_train, y_raw_singles_train, X_raw_singles_calibrate, y_raw_singles_calibrate = \
        split_data(X_raw_singles, y_raw_singles)

    if type_train_data == 'train':
        train_X = X_raw_singles_train
        train_y = y_raw_singles_train
    elif type_train_data == 'calibration':
        train_X = X_raw_singles_calibrate
        train_y = y_raw_singles_calibrate

    if retrain:
        # NB penile skin treated like all others for classify_single
        classify_single(train_X, train_y, inv_classes_map)

        model = MLPClassifier(random_state=0)
        # model = LogisticRegression(random_state=0)
        for n in range(n_folds):
            # TODO this is not nfold, but independently random
            X_train, X_test, y_train, y_test = train_test_split(train_X, train_y)
            while len(set(y_test)) != len(set(y_train)):
                # make sure we have all labels in both sets
                X_train, X_test, y_train, y_test = train_test_split(train_X, train_y)
            X_augmented_train, y_augmented_train, _, _ = augment_data(
                X_train,
                y_train,
                n_single_cell_types,
                n_features,
                N_SAMPLES_PER_COMBINATION,
                classes_map,
                from_penile=from_penile
            )

            print(
                'fitting on {} samples, {} features, {} classes'.format(
                    len(y_augmented_train),
                    X_augmented_train.shape[1],
                    len(set(y_augmented_train)))
            )

            # TODO get the mixture data from dorum

            model.fit(X_augmented_train, y_augmented_train)

            X_augmented_test, y_augmented_test, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
                X_test, y_test,
                n_single_cell_types,
                n_features,
                N_SAMPLES_PER_COMBINATION,
                classes_map,
                from_penile=from_penile
            )

            evaluate_model(
                model,
                'fold {}'.format(n),
                X_augmented_test,
                y_augmented_test,
                y_augmented_matrix,
                mixture_classes_in_single_cell_type,
                classes_map,
                MAX_LR
            )

            mixture_classes_in_classes_to_evaluate, _ = create_information_on_classes_to_evaluate(
                mixture_classes_in_single_cell_type,
                classes_map,
                class_combinations_to_evaluate,
                y_augmented_train,
                y_augmented_matrix
            )

            if n == 0:
                # only plot single class performance once
                boxplot_per_single_class_category(
                    X_augmented_test,
                    y_augmented_matrix,
                    classes_to_evaluate,
                    mixture_classes_in_classes_to_evaluate,
                    class_combinations_to_evaluate
                )

                # only make calibrated lrs once
                if type_train_data == 'calibration':
                    h1_h2_probs_calibration = evaluate_model(
                        model,
                        'fold {}'.format(n),
                        X_augmented_test,
                        y_augmented_test,
                        y_augmented_matrix,
                        mixture_classes_in_single_cell_type,
                        classes_map,
                        MAX_LR
                    )

                    # TODO: plot two separate histograms in one figure
                    #plot_histogram_log_lr()

                    bins = np.linspace(-10, 10, 30)
                    for i in range(len(h1_h2_probs_calibration)):
                        likrats1 = (h1_h2_probs_calibration[i][0] / (1 - h1_h2_probs_calibration[i][0]))
                        log_likrats1 = np.log10(likrats1)
                        likrats2 = (h1_h2_probs_calibration[i][1] / (1 - h1_h2_probs_calibration[i][1]))
                        log_likrats2 = np.log10(likrats2)

                        plt.hist([log_likrats1, log_likrats2], bins=bins, color=['pink', 'blue'],
                                 label=['h1', 'h2'])
                        plt.legend(loc='upper right')
                        plt.show()

                        # TODO: make reliability plot before KDE calibration
                        # Plot reliability plot
                        probabilities_before_calibration = np.append(h1_h2_probs_calibration[i][0],
                                                                     h1_h2_probs_calibration[i][1])
                        y_score_bin_mean, empirical_prob_pos = reliability_curve(
                            np.array(sorted(y_augmented_matrix[:, i], reverse=True)),
                            probabilities_before_calibration, bins=10)

                        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
                        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
                        plt.plot(y_score_bin_mean[scores_not_nan],
                                 empirical_prob_pos[scores_not_nan],
                                 label=str(i), color='red')
                        plt.ylabel("Empirical probability")
                        plt.legend(loc=0)


                    # TODO: make this feasible for all classes
                    X, y = Xn_to_Xy(h1_h2_probs_calibration[0][0], h1_h2_probs_calibration[0][1])
                    calibrator = KDECalibrator()
                    lr1, lr2 = Xy_to_Xn(calibrator.fit_transform(X, y), y)

                    # TODO: make reliability plot after KDE calibration

        # train on the full set and test on independent mixtures set
        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
            train_X,
            train_y,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

        model.fit(X_train, y_train)

        pickle.dump(model, open(model_file_name, 'wb'))
    else:
        model = pickle.load(open(model_file_name, 'rb'))
        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
            train_X,
            train_y,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

    evaluate_model(
        model,
        'train',
        X_train,
        y_train,
        y_augmented_matrix,
        mixture_classes_in_single_cell_type,
        classes_map,
        MAX_LR
    )

    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = read_mixture_data(
        n_single_cell_types - 1,
        n_features,
        classes_map
    )


    if retrain:
        X_augmented, y_augmented, _, _ = augment_data(
            train_X,
            train_y,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

        unique_augmented = np.unique(X_augmented, axis=0)
        dists_from_xmixtures_to_closest_augmented = []
        for x in tqdm(X_mixtures, 'computing distances'):
            dists_from_xmixtures_to_closest_augmented.append(np.min([np.linalg.norm(x - y) for y in unique_augmented]))
        pickle.dump(dists_from_xmixtures_to_closest_augmented, open('dists', 'wb'))
    else:
        dists_from_xmixtures_to_closest_augmented = pickle.load(open('dists', 'rb'))

    mixture_classes_in_classes_to_evaluate, y_mixtures_classes_to_evaluate_n_hot = \
        create_information_on_classes_to_evaluate(
            mixture_classes_in_single_cell_type,
            classes_map,
            class_combinations_to_evaluate,
            y_mixtures,
            y_mixtures_matrix
    )

    pickle.dump(mixture_classes_in_classes_to_evaluate, open('mixture_classes_in_classes_to_evaluate', 'wb'))

    h1_h2_probs_calibration = evaluate_model(
        model,
        'test mixtures',
        combine_samples(X_mixtures, n_features),
        y_mixtures,
        y_mixtures_classes_to_evaluate_n_hot,
        mixture_classes_in_classes_to_evaluate,
        classes_map,
        MAX_LR
    )

    plot_for_experimental_mixture_data(
        combine_samples(X_mixtures, n_features),
        y_mixtures,
        y_mixtures_classes_to_evaluate_n_hot,
        inv_test_map,
        classes_to_evaluate,
        mixture_classes_in_classes_to_evaluate,
        n_single_cell_types - 1,
        dists_from_xmixtures_to_closest_augmented
    )

    plot_calibration(h1_h2_probs_calibration, classes_to_evaluate)

    plt.close('all')
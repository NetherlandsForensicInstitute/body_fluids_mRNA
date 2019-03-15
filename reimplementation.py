import math
import pickle
import random
from collections import Counter, defaultdict

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
from lir.calibration import IsotonicCalibrator
from lir import pav

from scores import *


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

    X_raw = []
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
                n_discarded = 0

                end_replicate.append(len(rv_set_per_class))
                for i in range(n_full_samples):
                    if i == 0:
                        candidate_samples = data_for_this_label[:end_replicate[i], :]
                    else:
                        candidate_samples = data_for_this_label[
                                            end_replicate[i-1]:end_replicate[i], :
                                            ]
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


def combine_samples(data_for_class):
    """
    Combines the repeated measurements for each sample.

    :param data_for_class: N_samples x N_observations_per_sample x N_markers measurements numpy array
    :return: N_samples x N_markers measurements numpy array
    """
    data_for_class_mean = np.array([np.mean(data_for_class[i], axis=0)
                                    for i in range(data_for_class.shape[0])])
    return data_for_class_mean


def classify_single(X, y, inv_classes_map):
    """
    Very simple analysis of single cell type classification, useful as
    preliminary test.
    """
    # classify single classes
    single_samples = combine_samples(X)
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
    the shuffled replicates.

    :param X: N_single_cell_experimental_samples array and within a list filled with
        for each n_single_cell_experimental_sample a N_measurements per sample x N_markers array
    :param y: list of length N_single_cell_experimental_samples filled with int labels of which
        cell type was measured
    :param n: number of samples to generate
    :param classes_to_include: iterable of int, cell type indices to include
        in the mixtures
    :param n_features: int N_markers (=N_features)
    :return: n x n_features array
    """
    if len(classes_to_include) == 0:
        return np.zeros((n, n_features))
    data_for_class=[]
    for j, clas in enumerate(classes_to_include):
        data_for_class.append(X[np.argwhere(np.array(y) == clas).flatten()])

    augmented_samples = []
    for i in range(n):
        sampled = []
        for j, clas in enumerate(classes_to_include):

            n_in_class = sum(np.array(y) == clas)
            sampled_sample = data_for_class[j][np.random.randint(n_in_class)]
            n_replicates = len(sampled_sample)
            sampled.append(sampled_sample[np.random.permutation(n_replicates)])
        # TODO thus lower replicates for more cell types. is this an issue?
        smallest_replicates = min([len(sample) for sample in sampled])

        combined_sample = []
        for i_replicate in range(smallest_replicates):
            combined_sample.append(np.max(np.array([sample[i_replicate] for sample in sampled]), axis=0))

        augmented_samples.append(combined_sample)
    return combine_samples(np.array(augmented_samples))


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

    for i in range(2 ** n_single_cell_types_not_penile):
        binary = bin(i)[2:]
        while len(binary) < n_single_cell_types_not_penile:
            binary = '0' + binary
        classes_in_current_mixture = []
        for j, celltype in enumerate(sorted(classes_map)):
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
        cell_type = list(classes_map.keys())[list(classes_map.values()).index(j)]
        # get the probability per single class sample
        total_proba = prob_per_class[:, j]
        if sum(total_proba) > 0:
            probas_without_cell_type = total_proba[y_n_hot[:, j] == 0]
            probas_with_cell_type = total_proba[y_n_hot[:, j] == 1]
            # print(inv_classes_map[j], np.quantile(probas_without_cell_type, [0.05, .25, .5, .75, .95]),
            #       np.quantile(probas_with_cell_type, [0.05, .25, .5, .75, .95]))
            h1_h2_probs_per_class[cell_type] = (probas_with_cell_type, probas_without_cell_type)

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
    # TODO: Change the calculation of y_prob
    n_single_classes_to_draw = y_augmented_matrix.shape[1]
    y_prob = model.predict_proba(X_augmented_test)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
        y_prob, mixtures_in_classes_of_interest, classes_map_updated, MAX_LR)
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
        y_prob, mixtures_in_classes_of_interest, classes_map_updated, MAX_LR)

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

# TODO: check if function still needed
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
        X = combine_samples(X)

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
    plt.matshow(combine_samples(X))
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
    mixture_classes_in_classes_to_evaluate = mixture_classes_in_single_cell_type.copy()
    y_combi = np.zeros((len(y_mixtures), len(class_combinations_to_evaluate)))
    classes_map_updated = classes_map.copy()
    for i_combination, combination in enumerate(class_combinations_to_evaluate):
        labels = []
        str_combination = ''
        for k, cell_type in enumerate(combination):
            labels += mixture_classes_in_single_cell_type[cell_type]
            if k == 0:
                str_combination += cell_type + ' and/or '
            else:
                str_combination += cell_type
        mixture_classes_in_classes_to_evaluate[str_combination] = (list(set(labels)))
        classes_map_updated[str_combination] = len(classes_map_updated)
        for i in set(labels):
            # TODO: Does it have to be y_mixtures rather than y_test
            y_combi[np.where(np.array(y_mixtures) == i), i_combination] = 1

    return mixture_classes_in_classes_to_evaluate, classes_map_updated, \
           np.append(y_mixtures_matrix, y_combi, axis=1)


def split_data(X, y, size=(0.4, 0.4)):

    def indices_per_class(y):
        index_classes = list(np.unique(y, return_index=True)[1])[1:]
        index_classes1 = index_classes.copy()
        index_classes1.insert(0, 0)
        index_classes2 = index_classes.copy()
        index_classes2.append(len(y))
        index_classes = zip(index_classes1, index_classes2)

        indices_classes = [[i for i in range(index_class[0], index_class[1])] for index_class in index_classes]
        return indices_classes


    def define_random_indices(indices, size):
        train_size = size[0]
        calibration_size = size[1]

        train_index = random.sample(indices, int(train_size * len(indices)))
        indices_no_train = [idx for idx in indices if idx not in train_index]
        calibration_index = random.sample(indices_no_train, int(calibration_size * len(indices)))
        test_index = [idx for idx in indices if idx not in calibration_index
                      if idx not in train_index]

        return train_index, calibration_index, test_index

    # TODO: invoegen try-catch
    if sum(size) > 1.0:
        print("The sum of the sizes for the train and calibration"
              "data must be must be equal to or below 1.0.")

    indices_classes = indices_per_class(y)

    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = ([] for i in range(6))

    for indices_class in indices_classes:
        indices = [i for i in range(len(indices_class))]
        train_index, calibration_index, test_index = define_random_indices(indices, size)

        X_for_class = X[indices_class]
        y_for_class = [y[index_class] for index_class in indices_class]

        X_train.extend(X_for_class[train_index])
        y_train.extend([y_for_class[k] for k in train_index])
        X_calibrate.extend(X_for_class[calibration_index])
        y_calibrate.extend([y_for_class[k] for k in calibration_index])
        X_test.extend(X_for_class[test_index])
        y_test.extend([y_for_class[k] for k in test_index])

    print("The actual distribution (train, calibration, test) is ({}, {}, {})".format(
        len(X_train)/X.shape[0],
        len(X_calibrate)/X.shape[0],
        len(X_test)/X.shape[0]
    ))

    X_train = np.array(X_train)
    X_calibrate = np.array(X_calibrate)
    X_test = np.array(X_test)

    return X_train, y_train, X_calibrate, y_calibrate, X_test, y_test


if __name__ == '__main__':
    developing = False
    include_blank = False
    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map, inv_classes_map, n_per_class = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)
    # TODO: Make this function work
    #plot_data(X_raw_singles)
    n_folds = 2
    N_SAMPLES_PER_COMBINATION = 50
    MAX_LR = 10
    from_penile = False
    retrain = True
    model_file_name = 'mlpmodel'
    if from_penile:
        model_file_name+='_penile'

    # which classes should we compute marginals for? all single cell types and a 'contains vaginal' class?
    # '-1' to avoid the penile skin
    single_cell_classes = [inv_classes_map[j] for j in range(n_single_cell_types - 1)]
    class_combinations_to_evaluate = [['Vaginal.mucosa', 'Menstrual.secretion']]
    classes_to_evaluate = single_cell_classes + [' and/or '.join(comb) for comb in class_combinations_to_evaluate]

    # Split the data in two equal parts: for training and calibration
    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
        split_data(X_raw_singles, y_raw_singles, size=(0.4, 0.4))

    if retrain:
        # NB penile skin treated like all others for classify_single
        classify_single(X_train, y_train, inv_classes_map)

        model_scores = ScoresMLP()
        # model = MLPClassifier(random_state=0)
        # model = LogisticRegression(random_state=0)
        for n in range(n_folds):
            # TODO this is not nfold, but independently random
            X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
                split_data(X_raw_singles, y_raw_singles, size=(0.4, 0.4))

            # augment the train part of the data to train the MLP model on
            X_augmented_train, y_augmented_train, _, _ = \
                augment_data(
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
            model_scores.fit(X_augmented_train, y_augmented_train)

            # augment calibration data to calibrate the model with
            X_calibrate_augmented, y_calibrate_augmented, y_augmented_matrix_calibrate, mixture_classes_in_single_cell_type = \
                augment_data(
                    X_calibrate,
                    y_calibrate,
                    n_single_cell_types,
                    n_features,
                    N_SAMPLES_PER_COMBINATION,
                    classes_map,
                    from_penile=from_penile
                )

            h1_h2_probs_calibration = model_scores.predict_proba_per_class(
                X_calibrate_augmented,
                y_augmented_matrix_calibrate,
                mixture_classes_in_single_cell_type,
                classes_map,
                MAX_LR
            )

            # augment test data to evaluate the model with
            X_augmented_test, y_augmented_test, y_augmented_matrix, mixture_classes_in_single_cell_type = \
                augment_data(
                    X_test,
                    y_test,
                    n_single_cell_types,
                    n_features,
                    25,
                    classes_map,
                    from_penile=from_penile
            )

            # np.sort(y_augmented_matrix, axis=1)

            mixture_classes_in_classes_to_evaluate, classes_map_updated, _ = create_information_on_classes_to_evaluate(
                mixture_classes_in_single_cell_type,
                classes_map,
                class_combinations_to_evaluate,
                y_augmented_test,
                y_augmented_matrix
            )

            # TODO: Change the classes_map if want to evaluate multiple classes
            h1_h2_probs_test = model_scores.predict_proba_per_class(
                X_augmented_test,
                y_augmented_matrix,
                mixture_classes_in_single_cell_type,
                classes_map,
                MAX_LR
            )



            # fit calibrated models
            calibrators_per_class = calibration_fit(h1_h2_probs_calibration, classes_map)

            # transform the test scores
            h1_h2_after_calibration = calibration_transform(h1_h2_probs_test, calibrators_per_class, classes_map)

            if n == 0:
                # only plot single class performance once
                # TODO: Make this function work
                # boxplot_per_single_class_category(
                #     X_augmented_test,
                #     y_augmented_matrix,
                #     classes_to_evaluate,
                #     mixture_classes_in_classes_to_evaluate,
                #     class_combinations_to_evaluate
                # )

                # plots before calibration making use of probabilities
                plot_histogram_log_lr(h1_h2_probs_test, title='before')
                plot_reliability_plot(h1_h2_probs_test, y_augmented_matrix, title='before')

                # plots after calibration making use of probabilities
                plot_histogram_log_lr(h1_h2_after_calibration, title='after')
                plot_reliability_plot(h1_h2_after_calibration, y_augmented_matrix, title='after')

                # make plots before calibration making use of loglrs
                h1_h2_lrs_test = {}
                for celltype in sorted(classes_map):
                    h1_h2_celltype = h1_h2_probs_test[celltype]
                    h1_h2_lrs_test[celltype] = [h1_h2_celltype[i] / (1 - h1_h2_celltype[i]) for i in
                                                range(len(h1_h2_celltype))]

                # TODO: Check y_augmented matrix and what happens with h1_h2_probs why equally divided?
                y_matrix_test = np.append(np.ones((3200, 8)), np.zeros((3200, 8)), axis=0)
                y = y_matrix_test[:, 0] # list with two class labels

                # make plots before calibration making use of loglrs
                h1_h2_lrs_after_calibration = {}
                for celltype in sorted(classes_map):
                    h1_h2_celltype_calib = h1_h2_after_calibration[celltype]
                    h1_h2_lrs_after_calibration[celltype] = [h1_h2_celltype_calib[i] / (1 - h1_h2_celltype_calib[i]) for i in
                                                range(len(h1_h2_celltype_calib))]

                y_matrix_test_c = np.append(np.ones((3200, 8)), np.zeros((3200, 8)), axis=0)
                y_c = y_matrix_test_c[:, 0] # list with two class labels

                for celltype in sorted(classes_map):
                    lrs = np.append(h1_h2_lrs_test[celltype][0],
                                    h1_h2_lrs_test[celltype][1])

                    pav.plot(lrs, y, on_screen=True)

                    lrs_c = np.append(h1_h2_lrs_after_calibration[celltype][0],
                                      h1_h2_lrs_after_calibration[celltype][1])

                    pav.plot(lrs_c, y, on_screen=True)


        X_augmented_train, y_augmented_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
            X_train,
            y_train,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

        X_augmented_calibrate, y_augmented_calibrate, y_augmented_matrix_calibrate, mixture_classes_in_single_cell_type = \
            augment_data(
                X_calibrate,
                y_calibrate,
                n_single_cell_types,
                n_features,
                N_SAMPLES_PER_COMBINATION,
                classes_map,
                from_penile=from_penile
            )

        X_augmented_test, y_augmented_test, y_augmented_matrix, mixture_classes_in_single_cell_type = \
            augment_data(
                X_test,
                y_test,
                n_single_cell_types,
                n_features,
                25,
                classes_map,
                from_penile=from_penile
            )

        model_scores.fit(X_augmented_train, y_augmented_train)

        h1_h2_probs_test = model_scores.predict_proba_per_class(
            X_augmented_test,
            y_augmented_matrix,
            mixture_classes_in_single_cell_type,
            classes_map,
            MAX_LR
        )

        calibrators_per_class = calibration_fit(h1_h2_probs_test, classes_map)

        pickle.dump(model_scores, open(model_file_name, 'wb'))
        pickle.dump(calibrators_per_class, open('calibrators_per_class', 'wb'))
    else:
        model_scores = pickle.load(open(model_file_name, 'rb'))
        calibrators_per_class = pickle.load(open('calibrators_per_class', 'rb'))

        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
            X_train,
            y_train,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

    # calculate the probs from test data with the MLP model trained on train data
    # evaluate_model(
    #     model_scores,
    #     'train',
    #     X_train,
    #     y_train,
    #     y_augmented_matrix,
    #     mixture_classes_in_single_cell_type,
    #     classes_map,
    #     MAX_LR
    # )å

    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = read_mixture_data(
        n_single_cell_types - 1,
        n_features,
        classes_map
    )

    if retrain:
        X_augmented, y_augmented, _, _ = augment_data(
            X_train,
            y_train,
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

    mixture_classes_in_classes_to_evaluate, classes_map_updated, y_mixtures_classes_to_evaluate_n_hot = \
        create_information_on_classes_to_evaluate(
            mixture_classes_in_single_cell_type,
            classes_map,
            class_combinations_to_evaluate,
            y_mixtures,
            y_mixtures_matrix
    )

    h1_h2_probs_mixture = model_scores.predict_proba_per_class(
        combine_samples(X_mixtures),
        y_mixtures_classes_to_evaluate_n_hot,
        mixture_classes_in_classes_to_evaluate,
        classes_map_updated,
        MAX_LR
    )

    # transform the probabilities with the calibrated models
    # TODO: How to take into account the combined classes (e.g. vaginal + menstrual)?
    h1_h2_after_calibration_mixture = calibration_transform(h1_h2_probs_mixture, classes_map_updated)

    plot_for_experimental_mixture_data(
        combine_samples(X_mixtures),
        y_mixtures,
        y_mixtures_classes_to_evaluate_n_hot,
        inv_test_map,
        classes_to_evaluate,
        mixture_classes_in_classes_to_evaluate,
        n_single_cell_types - 1,
        dists_from_xmixtures_to_closest_augmented
    )

    plot_calibration(h1_h2_after_calibration_mixture, classes_to_evaluate)
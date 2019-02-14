import math
import pickle
from collections import Counter, defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
# from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from metrics import get_lr_metrics


def read_df(filename, binarize):
    """
    reads in an xls file as a dataframe, replacing NA and binarizing if required
    :param filename: name of file to read in
    :param binarize: whether to binarize - use a cutoff value to convert to 0/1
    :return: dataframe
    """
    df = pd.read_excel(filename)
    df.fillna(0, inplace=True)
    if binarize:
        df = 1 * (df > 150)
    return df


def get_data_per_cell_type(filename='Datasets/Dataset_NFI.xlsx', binarize=True, developing=False, include_blank=False):
    """
    returns data per specified cell types

    :param filename: name of file to read in
    :param binarize: whether to binarize raw measurement values
    :param developing: if developing, ignore Skin and Blank category for speed
    :param include_blank: whether to include Blank as a separate cell type
    :return: (N_single_cell_experimental_samples x N_measurements per sample x N_markers array of measurements,
                N_single_cell_experimental_samples array of int labels of which cell type was measured,
                N_cell types,
                N_markers (=N_features),
                dict: cell type name -> cell type index,
                dict: cell type index -> cell type name,
                dict: cell type index -> N_measurements for cell type

    """
    df = read_df(filename, binarize)
    # restructure data
    classes_labels = np.array(df.index)
    # penile skin should be treated separately
    classes_set = set(classes_labels)
    classes_set.remove('Skin.penile')
    classes_map = {}
    inv_classes_map = {}
    i = 0
    for clas in sorted(classes_set) + ['Skin.penile']:
        if include_blank or 'Blank' not in clas:
            if not developing or ('Skin' not in clas and 'Blank' not in clas):
                classes_map[clas] = i
                inv_classes_map[i] = clas
                i += 1
    classes = [classes_map[clas] for clas in classes_labels if
               (not developing or ('Skin' not in clas and 'Blank' not in clas))
               and (include_blank or 'Blank' not in clas)]
    n_per_class = Counter(classes)
    n_features = len(df.columns)
    X_raw = np.zeros([0, 4, n_features])
    y = []
    for clas in sorted(classes_set) + ['Skin.penile']:
        if include_blank or 'Blank' not in clas:
            if not developing or ('Skin' not in clas and 'Blank' not in clas):
                full_set_per_class = np.array(df.loc[clas])
                # TODO find which make pairs of 4
                # assuming they are ordered 4 together
                n_samples, n_features = full_set_per_class.shape
                print(clas, n_samples / 4)
                # for now, discard the remainder?
                n_full_samples = int(n_samples / 4)
                n_per_class[classes_map[clas]] = n_full_samples
                data_for_class = np.zeros((n_full_samples, 4, n_features))
                n_discarded = 0
                for i in range(n_full_samples):
                    candidate_samples = full_set_per_class[i * 4:(i + 1) * 4, :]
                    # discard if the structural measurements were empty
                    if sum(candidate_samples[:, -1]) < 3 or sum(candidate_samples[:, -2]) < 3 and 'Blank' not in clas:
                        n_full_samples -= 1
                        data_for_class = data_for_class[:n_full_samples, :, :]
                        n_discarded += 1
                    else:
                        data_for_class[i - n_discarded, :, :] = candidate_samples

                print('{} has {} samples (after discarding {} due to QC on structural markers)'.format(clas,
                                                                                                       n_full_samples,
                                                                                                       n_discarded))
                X_raw = np.append(X_raw, data_for_class, axis=0)
                y += [classes_map[clas]] * n_full_samples
    return X_raw, y, len(classes_map), n_features, classes_map, inv_classes_map, n_per_class


def combine_samples(data_for_class):
    """
    takes a n_samples x 4 x n_features matrix and returns the n_samples x n_markers matrix
    :param data_for_class:
    :return: n_samples x N_markers array
    """
    return np.mean(data_for_class, axis=1)


def classify_single(X, y, inv_classes_map):
    """
    very simple analysis of single cell type classification, useful as preliminary test
    """
    # classify single classes
    single_samples = combine_samples(X)
    print('fitting on {} samples, {} features, {} classes'.format(len(y), single_samples.shape[1],

                                                                  len(set(y))))

    X_train, X_test, y_train, y_test = train_test_split(single_samples, y)
    single_model = MLPClassifier(random_state=0)
    single_model.fit(X_train, y_train)
    y_pred = single_model.predict(X_test)
    print('train accuracy for single classes: {}'.format(accuracy_score(y_test, y_pred)))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print(cnf_matrix)
    print(inv_classes_map)



def construct_random_samples(X, y, n, classes_to_include, n_features):
    """
    returns n generated samples that contain classes classes_to_include.
    A sample is generated by random sampling a sample for each class, and adding the shuffled replicates
    :param X: N_single_cell_experimental_samples x N_measurements per sample x N_markers array of measurements
    :param y: N_single_cell_experimental_samples array of int labels of which cell type was measured
    :param n: int: number of samples to generate
    :param classes_to_include:  iterable of int, cell type indices to include in the mixtures
    :param n_features: int N_markers (=N_features),
    :return n x n_features array
    """
    if len(classes_to_include) == 0:
        return np.zeros((n, n_features))
    sampled = np.zeros((len(classes_to_include), n, 4, n_features))
    for j, clas in enumerate(classes_to_include):
        n_in_class = sum(np.array(y) == clas)
        data_for_class = np.array([X[i, :, :] for i in range(len(X)) if y[i] == clas])
        sampled[j, :, :, :] = data_for_class[np.random.randint(n_in_class, size=n), :, :]
        # shuffle them
        for i in range(n):
            sampled[j, i, :, :] = sampled[j, i, np.random.permutation(4), :]
    combined = np.max(sampled, axis=0)
    return combine_samples(combined)


def augment_data(X_singles_raw, y_singles, n_single_cell_types, n_features, from_penile=False):
    """
    Generate data for the power set of single cell types

    :param X_singles_raw: N_single_cell_experimental_samples x N_measurements per sample x N_markers array of measurements
    :param y_singles: N_single_cell_experimental_samples array of int labels of which cell type was measured
    :param n_single_cell_types: int: number of single cell types
    :param n_features: int: N_markers
    :param from_penile: bool: generate samplew that (T) always or (F) never also contain penile skin
    :return: N_experiments x N_markers array,
                N_experiment array of int labels for the powerset (=mixtures) classes,
                N_augmented_data_samples x N_single_cell_classes matrix of 0, 1, indicating for each augmented sample
                            which single cell types it was made up of. Does not contain a column for penile skin,
                list of length N_single_cell_classes of lists, that indicate the mixture labels each single cell type
                            features in
    """
    # generate more data
    X = np.zeros((0, n_features))
    y = []
    n_single_cell_types_not_penile = n_single_cell_types - 1
    y_n_hot = np.zeros((2 ** n_single_cell_types_not_penile * N_SAMPLES_PER_COMBINATION, n_single_cell_types),
                       dtype=int)
    mixtures_containing_single_cell_type = [[] for _ in range(n_single_cell_types_not_penile)]
    for i in range(2 ** n_single_cell_types_not_penile):
        binary = bin(i)[2:]
        while len(binary) < n_single_cell_types_not_penile:
            binary = '0' + binary
        classes_in_current_mixture = []
        for j in range(n_single_cell_types_not_penile):
            if binary[-j - 1] == '1':
                classes_in_current_mixture.append(j)
                mixtures_containing_single_cell_type[j].append(int(i))
                y_n_hot[i * N_SAMPLES_PER_COMBINATION:(i + 1) * N_SAMPLES_PER_COMBINATION, j] = 1
        if from_penile:
            # also (always) add penile skin samples
            y_n_hot[i * N_SAMPLES_PER_COMBINATION:(i + 1) * N_SAMPLES_PER_COMBINATION, n_single_cell_types - 1] = 1
        X = np.append(X, construct_random_samples(X_singles_raw, y_singles, N_SAMPLES_PER_COMBINATION,
                                                  classes_in_current_mixture,
                                                  n_features), axis=0)
        y += [i] * N_SAMPLES_PER_COMBINATION
    return X, y, y_n_hot[:, :n_single_cell_types_not_penile], mixtures_containing_single_cell_type


def evaluate_model(model, dataset_label, X, y, y_n_hot, labels_in_class):
    """
    computes metrics for performance of the model on dataset X, y

    :param model: sklearn-like model to evaluate
    :param dataset_label:
    :param X:
    :param y:
    :param y_n_hot:
    :param labels_in_class:
    :return: iterable with for each class a list of len 2, with scores for all h1 and h2 scenarios
    """
    y_pred = model.predict(X)
    print('{} accuracy for mixtures: {}'.format(dataset_label, accuracy_score(y, y_pred)))
    y_prob = model.predict_proba(X)
    scores_per_class = {}
    # marginal for each single class sample
    prob_per_class = convert_prob_per_mixture_to_marginal_per_class(y_prob, labels_in_class)
    for j in range(y_n_hot.shape[1]):
        total_proba = prob_per_class[:, j]
        if sum(total_proba) > 0:
            probas_without_cell_type = total_proba[y_n_hot[:, j] == 0]
            probas_with_cell_type = total_proba[y_n_hot[:, j] == 1]
            # print(inv_classes_map[j], np.quantile(probas_without_cell_type, [0.05, .25, .5, .75, .95]),
            #       np.quantile(probas_with_cell_type, [0.05, .25, .5, .75, .95]))
            scores_per_class[j] = (probas_with_cell_type, probas_without_cell_type)
    return scores_per_class


def convert_prob_per_mixture_to_marginal_per_class(prob, labels_in_class):
    """
    converts n_samples x n_mixture_classes matrix of probabilities to a n_samples x n_classes_of_interest matrix, by
    summing over the relevant mixtures

    :param prob: n_samples x n_mixture_classes matrix of probabilities
    :param labels_in_class: iterable of len n_classes_of_interest. For each class, the list of mixture classes that contain
    the class of interest are given
    :return: n_samples x n_classes_of_interest matrix of probabilities
    """
    res_prob = np.zeros((prob.shape[0], len(labels_in_class)))
    for j in range(res_prob.shape[1]):
        if len(labels_in_class[j]) > 0:
            res_prob[:, j] = np.sum(prob[:, labels_in_class[j]], axis=1)
    epsilon = 10 ** -MAX_LR
    res_prob = np.where(res_prob > 1 - epsilon, 1 - epsilon, res_prob)
    res_prob = np.where(res_prob < epsilon, epsilon, res_prob)
    return res_prob


def read_mixture_data(n_single_cell_types_no_penile, binarize=True):
    """
    reads in the experimental mixture data that is used as test data
    :param n_single_cell_types_no_penile: int: number of single cell types excluding penile skine
    :param binarize: bool: whether to binarize values
    :return: N_samples x N_markers array of measurements NB only one replicate per sample,
                N_samples iterable of mixture class labels - corresponds to the labels used in data augmentation,
                N_samples x N_single_cell_type n_hot encoding of the labels NB in in single cell type space!
                dict: mixture name -> list of int single cell type labels
                dict: mixture class label -> mixture name
    """
    # read test data
    df = read_df('Datasets/Dataset_mixtures.xlsx', binarize)
    # restructure data
    # TODO this is one per replicate currently! - are there replications?
    test_labels = np.array(df.index)
    test_map = defaultdict(list)
    X_mixtures = np.zeros((0, n_features))
    y_mixtures = []
    inv_test_map = {}
    y_mixtures_n_hot = np.zeros((len(df), n_single_cell_types_no_penile), dtype=int)
    n_total = 0
    for test_label in sorted(set(test_labels)):
        labels = test_label.split('+')
        class_label = 0
        data_for_this_label = np.array(df.loc[test_label], dtype=float)
        n = data_for_this_label.shape[0]
        for label in labels:
            test_map[test_label].append(classes_map[label])
            class_label += 2 ** classes_map[label]
            y_mixtures_n_hot[n_total:n_total + n, classes_map[label]] = 1
        inv_test_map[class_label] = test_label
        n_total += n
        X_mixtures = np.append(X_mixtures, data_for_this_label, axis=0)
        y_mixtures += [class_label] * data_for_this_label.shape[0]
    return X_mixtures, y_mixtures, y_mixtures_n_hot, test_map, inv_test_map


def boxplot_per_single_class_category(X_augmented_test, y_augmented_matrix, classes_to_evaluate,
                                      mixtures_in_classes_of_interest, class_combinations_to_evaluate):
    """
    for single cell type, plot the distribution of marginal LRs for each cell type, as well as for specified
    combinations of classes

    :param X_augmented_test: N_samples x N_markers array of observations
    :param y_augmented_matrix: N_samples x (N_single_cell_types + N_combos) n_hot encoding
    :param classes_to_evaluate: list of str, names of classes to evaluate
    :param mixtures_in_classes_of_interest: list of lists, specifying for each class in classes_to_evaluate which
    mixture labels contain these
    :param class_combinations_to_evaluate: list of lists of int, specifying combinations of single cell types to consider
    :return: None
    """
    n_single_classes_to_draw = y_augmented_matrix.shape[1]
    y_prob = model.predict_proba(X_augmented_test)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(y_prob, mixtures_in_classes_of_interest)
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


def plot_for_experimental_mixture_data(X_mixtures, y_mixtures, y_mixtures_matrix, inv_test_map, classes_to_evaluate,
                                       mixtures_in_classes_of_interest,
                                       n_single_cell_types_no_penile, dists):
    """
    for each mixture category that we have measurements on, plot the distribution of marginal LRs for each cell type,
    as well as for the special combinations (eg vaginal+menstrual)
    also plot LRs as a function of distance to nearest data point
    also plot experimental measurements together with LRs found and distance in a large matrix plot

    :param X_mixtures: N_experimental_mixture_samples x N_markers array of observations
    :param y_mixtures: N_experimental_mixture_samples array of int mixture labels
    :param y_mixtures_matrix:  N_experimental_mixture_samples x (N_single_cell_types + N_combos) n_hot encoding
    :param inv_test_map: dict: mixture label -> mixture name
    :param classes_to_evaluate: list of str, classes to evaluate
    :param mixtures_in_classes_of_interest:  list of lists, specifying for each class in classes_to_evaluate which
    mixture labels contain these
    :param n_single_cell_types_no_penile: int: number of single cell types excluding penile skin
    :param dists: N_experimental_mixture_samples iterable of distances to nearest augmented data point. Indication of
            whether the point may be an outlier (eg measurement error or problem with augmentation scheme)
    """
    # This is a test comment
    y_prob = model.predict_proba(X_mixtures)
    y_prob_per_class = convert_prob_per_mixture_to_marginal_per_class(y_prob, mixtures_in_classes_of_interest)

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
                for comb_class in cla.split(' and/or '):
                    if comb_class in inv_test_map[i_clas]:
                        patch.set_facecolor('black')
        plt.title(inv_test_map[i_clas])
    plt.savefig('mixtures_boxplot')

    plt.subplots(3, 3, figsize=(18, 9))
    for i in range(y_mixtures_matrix.shape[1]):
        plt.subplot(3, 3, i + 1)
        plt.ylim([-MAX_LR - .5, MAX_LR + .5])
        plt.scatter(dists + np.random.random(len(dists)) / 20, log_lrs_per_class[:, i],
                    color=['red' if iv else 'blue' for iv in y_mixtures_matrix[:, i]], alpha=0.1)
        plt.ylabel('LR')
        plt.xlabel('distance to nearest data point')
        plt.title(classes_to_evaluate[i])
    plt.savefig('LRs_as_a_function_of_distance')

    plt.figure()
    plt.matshow(
        np.append(np.append(X_mixtures, log_lrs_per_class, axis=1), np.expand_dims(np.array([d*5 for d in dists]), axis=1), axis=1))
    plt.savefig('mixtures binned data and log lrs')


def plot_data(X):
    """
    plots the raw data points

    :param X: N_samples x N_observations_per_sample x N_markers measurements
    """
    plt.matshow(combine_samples(X))
    plt.savefig('single_cell_type_measurements_after_QC')


def plot_calibration(h1_h2_scores, classes_to_evaluate):
    """
    print metrics on and generate plots on calibration NB the confidence intervals appear to still have issues
    :param h1_h2_scores: iterable with for each class to evaluate a list of len two, containing scores for h1 and h2
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
            m = get_lr_metrics(h1_scores=h1_scores, h2_scores=h2_scores, h1_lrs=h1_lrs, h2_lrs=h2_lrs, hp_prior=0.5)
            print(classes_to_evaluate[j], ['{}: {}'.format(a[0], round(a[1], 2)) for a in m])
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
            plt.bar(np.array(bins1_x) + .15, bins1_y, label='h1 (1)', width=.3, color='red')
            if len(std_err1) > 0:
                bins_se1_x, vals = zip(*sorted(std_err1.items()))
                bins_se1_y, y1 = zip(*vals)
                plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y, color='red')
        # plt.legend()
        plt.title(classes_to_evaluate[j])
    plt.savefig('calibration separate')

    plt.figure()
    all_std_err0, all_bins0 = transform_counts(all_bins0, sum([len(b[1]) for a, b in h1_h2_scores.items()]), scale,
                                               True)

    all_std_err1, all_bins1 = transform_counts(all_bins1, sum([len(b[0]) for a, b in h1_h2_scores.items()]), scale,
                                               False)

    bins0_x, bins0_y = zip(*sorted(all_bins0.items()))
    plt.bar(np.array(bins0_x) - .15, bins0_y, label='h2 (0)', width=.3)
    if len(all_std_err0) > 0:
        bins_se1_x, vals = zip(*sorted(all_std_err0.items()))
        bins_se1_y, y1 = zip(*vals)
        plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y)
    bins1_x, bins1_y = zip(*sorted(all_bins1.items()))
    plt.bar(np.array(bins1_x) + .15, bins1_y, label='h1 (1)', width=.3, color='red')
    if len(all_std_err1) > 0:
        bins_se1_x, vals = zip(*sorted(all_std_err1.items()))
        bins_se1_y, y1 = zip(*vals)
        plt.errorbar(np.array(bins_se1_x) + .15, y1, yerr=bins_se1_y, color='red')
    plt.legend()
    plt.title('all')
    plt.savefig('calibration all')


def transform_counts(bins, n_obs, scale, is_h2):
    """
    transform counts so h1 and h2 fractions can be visually compared
    if the score is 'correct' (ie log > 0 for h1 and < 0 for h2), just take the fraction
    if the score is 'incorrect', multiply by how much more often the score should occur in the 'correct' scenario, ie by
    10**the value of the score
    also provides an (apparently incorrect) standard error for each bin
    :param bins: dict: rounded score -> count
    :param n_obs: int: total number of observations
    :param scale: logscale, eg 10
    :param is_h2: whether these are the counts for h2 (ie False -> h1)
    :return: dict: rounded score -> (standard error * adjustment factor, adjustment factor),
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


def create_information_on_classes_to_evaluate(mixture_classes_in_single_cell_type, classes_map,
                                              class_combinations_to_evaluate, y_mixtures, y_mixtures_matrix):
    """
    generates data structures pertaining to all classes to evaluate, which are single cell types and certain
    combinations thereof
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
            labels += mixture_classes_in_single_cell_type[classes_map[cell_type]]
        mixture_classes_in_classes_to_evaluate.append(list(set(labels)))
        for i in set(labels):
            y_combi[np.where(np.array(y_mixtures) == i), i_combination] = 1

    return mixture_classes_in_classes_to_evaluate, np.append(y_mixtures_matrix, y_combi, axis=1)


### Naomi's part ###
def read_original_data(filename):
    xls = pd.ExcelFile(filename)

    sheet_to_df_map = {}
    for sheet_name in xls.sheet_names:
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)

    return sheet_to_df_map

def getNumeric(prompt):
    while True:
        response = input(prompt)
        try:
            return int(response)
        except ValueError:
            print("Please enter a number.")

if __name__ == '__main__':
    developing = False
    include_blank = False
    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map, inv_classes_map, n_per_class = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)
    plot_data(X_raw_singles)
    n_folds = 2
    N_SAMPLES_PER_COMBINATION = 100
    MAX_LR=10
    from_penile = False
    retrain = False
    model_file_name = 'mlpmodel'
    if from_penile:
        model_file_name+='_penile'

    # which classes should we compute marginals for? all single cell types and a 'contains vaginal' class?
    # '-1' to avoid the penile skin
    single_cell_classes = [inv_classes_map[j] for j in range(n_single_cell_types - 1)]
    class_combinations_to_evaluate = [['Vaginal.mucosa', 'Menstrual.secretion']]
    classes_to_evaluate = single_cell_classes + [' and/or '.join(comb) for comb in class_combinations_to_evaluate]

    if retrain:
        # NB penile skin treated like all others for classify_single
        classify_single(X_raw_singles, y_raw_singles,inv_classes_map)

        model = MLPClassifier(random_state=0)
        # model = LogisticRegression(random_state=0)
        for n in range(n_folds):
            # TODO this is not nfold, but independently random
            X_train, X_test, y_train, y_test = train_test_split(X_raw_singles, y_raw_singles)
            while len(set(y_test)) != len(set(y_train)):
                # make sure we have all labels in both sets
                X_train, X_test, y_train, y_test = train_test_split(X_raw_singles, y_raw_singles)
            X_augmented_train, y_augmented_train, _, _ = augment_data(X_train,
                                                                      y_train,
                                                                      n_single_cell_types, n_features,
                                                                      from_penile=from_penile)

            print(
                'fitting on {} samples, {} features, {} classes'.format(len(y_augmented_train),
                                                                        X_augmented_train.shape[1],

                                                                        len(set(y_augmented_train))))

            #  try calibration - or skip that?
            # TODO get the mixture data from dorum

            model.fit(X_augmented_train, y_augmented_train)

            X_augmented_test, y_augmented_test, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
                X_test, y_test,
                n_single_cell_types,
                n_features,
                from_penile=from_penile)

            evaluate_model(model, 'fold {}'.format(n), X_augmented_test, y_augmented_test, y_augmented_matrix,
                           mixture_classes_in_single_cell_type)

            mixture_classes_in_classes_to_evaluate, _ = create_information_on_classes_to_evaluate(
                mixture_classes_in_single_cell_type, classes_map, class_combinations_to_evaluate, y_augmented_train, y_augmented_matrix)

            if n == 0:
                # only plot single class performance once
                boxplot_per_single_class_category(X_augmented_test, y_augmented_matrix,
                                                  classes_to_evaluate,
                                                  mixture_classes_in_classes_to_evaluate,
                                                  class_combinations_to_evaluate)

        # train on the full set and test on independent mixtures set
        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(X_raw_singles,
                                                                                                 y_raw_singles,
                                                                                                 n_single_cell_types,
                                                                                                 n_features,
                                                                                                 from_penile=from_penile)

        model.fit(X_train, y_train)

        pickle.dump(model, open(model_file_name, 'wb'))
    else:
        model = pickle.load(open(model_file_name, 'rb'))
        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(X_raw_singles,
                                                                                                 y_raw_singles,
                                                                                                 n_single_cell_types,
                                                                                                 n_features,
                                                                                                 from_penile=from_penile)

    evaluate_model(model, 'train', X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type)

    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = read_mixture_data(n_single_cell_types - 1)

    X_augmented, y_augmented, _, _ = augment_data(X_raw_singles,
                                                  y_raw_singles,
                                                  n_single_cell_types, n_features,
                                                  from_penile=from_penile)

    if retrain:
        unique_augmenteds = np.unique(X_augmented, axis=0)
        dists_from_xmixtures_to_closest_augmented = []
        for x in tqdm(X_mixtures, 'computing distances'):
            dists_from_xmixtures_to_closest_augmented.append(np.min([np.linalg.norm(x - y) for y in unique_augmenteds]))
        pickle.dump(dists_from_xmixtures_to_closest_augmented, open('dists', 'wb'))
    else:
        dists_from_xmixtures_to_closest_augmented = pickle.load(open('dists', 'rb'))

    mixture_classes_in_classes_to_evaluate, y_mixtures_classes_to_evaluate_n_hot = create_information_on_classes_to_evaluate(
        mixture_classes_in_single_cell_type, classes_map, class_combinations_to_evaluate, y_mixtures, y_mixtures_matrix)

    h1_h2_scores = evaluate_model(model, 'test mixtures', X_mixtures, y_mixtures,
                                  y_mixtures_classes_to_evaluate_n_hot,
                                  mixture_classes_in_classes_to_evaluate)

    plot_for_experimental_mixture_data(X_mixtures, y_mixtures, y_mixtures_classes_to_evaluate_n_hot,
                                       inv_test_map, classes_to_evaluate,
                                       mixture_classes_in_classes_to_evaluate, n_single_cell_types - 1,
                                       dists_from_xmixtures_to_closest_augmented)

    plot_calibration(h1_h2_scores, classes_to_evaluate)

    ### Naomi's part ###
    # Assign the correct replicates to the same sample for the single body fluids.
    sheet_to_df_map = read_original_data('Datasets/Dataset_NFI_adj.xlsx')
    relevant_column = list(zip(*list(sheet_to_df_map['Samples + details'].index)))[3]
    shortened_names = [relevant_column[i][-3:] for i in range(len(relevant_column))]

    replicate_values = OrderedDict()
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
        if shortened_names[i][-1] in ['0', '6']:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]

    # TODO: adjust the shortened_names
    # TODO: iterate over the index and manually adjust
    # if str --> 1
    # if ends with either 0.3, 0.75, 0.5 --> check
    # if ends with a 6 --> check
    # if ends with a 0 --> 1
    # if ends with _1 --> check

    k = getNumeric("Give the number of buckets:")

    for i in indexes_to_be_checked:
        print("The rowname in the excel file is:", relevant_column[i])


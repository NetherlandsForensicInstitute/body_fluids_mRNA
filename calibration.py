"""

Check whether calibration is needed and if so perform calibration.

"""

import pickle
import collections

import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter

from reimplementation import get_data_per_cell_type, read_mixture_data, \
    calculate_lrs, combine_samples

from lir.calibration import KDECalibrator

if __name__ == '__main__':
    MAX_LR = 10
    # get correct data
    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map,\
        inv_classes_map, n_per_class = get_data_per_cell_type()
    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = \
        read_mixture_data(n_single_cell_types - 1, n_features, classes_map=classes_map)

    mixture_classes_in_classes_to_evaluate = pickle.load(
        open('mixture_classes_in_classes_to_evaluate', 'rb'))

    model = pickle.load(open('mlpmodel', 'rb'))
    log_lrs_per_class = calculate_lrs(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=True)
    lrs_per_class = calculate_lrs(
        X_mixtures, model, mixture_classes_in_classes_to_evaluate, n_features, MAX_LR, log=False)

    # TODO: split the data in two parts --> train and calibration set
    # TODO: augment both sets
    # TODO: train the model
    # TODO: calculate the scores of the trained model
    # TODO: plot the log_lrs_per_class per sample: histograms
    n_per_mixture_class = collections.Counter(y_mixtures)
    unique_groups_mixt, indices = np.unique(y_mixtures, return_index=True)
    sorted_test_map = collections.OrderedDict(sorted(n_per_mixture_class.items()))
    n_observations_mixt = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]

    sorted_classes_map = collections.OrderedDict(sorted(classes_map.items(), key=itemgetter(1)))
    names_single = np.array([list(sorted_classes_map.keys())])

    sorted_test_map = collections.OrderedDict(sorted(inv_test_map.items()))
    names_mixt = np.array([list(sorted_test_map.values())]).flatten()[np.argsort(indices)]
    names_mixt = np.reshape(names_mixt, (7, 1))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']
    for j, j_class in enumerate(set(y_mixtures)):
        print(j_class)
        indices_experiments = [k for k in range(len(y_mixtures)) if y_mixtures[k] == j_class]
        plt.subplots(2, 5, figsize=(18, 9))
        plt.suptitle(sorted_test_map[j_class], y=1.0, fontsize=14)
        for i in range(log_lrs_per_class.shape[1]):
            plt.subplot(2, 5, i + 1)
            plt.xlim([-MAX_LR - .5, MAX_LR + .5])
            plt.hist(log_lrs_per_class[indices_experiments, i],
                     color=colors[j])
            plt.title(names_single[0][i])
        plt.tight_layout()
        plt.show()





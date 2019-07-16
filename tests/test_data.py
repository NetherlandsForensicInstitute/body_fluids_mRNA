"""
Compares augmented data based on taking the mean and on taking the sum.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from rna.augment import MultiLabelEncoder, augment_data
from rna.analytics import combine_samples
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import string2vec

N_SAMPLES_PER_COMBINATION = 4
mle = MultiLabelEncoder(len(single_cell_types))

X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
    get_data_per_cell_type(single_cell_types=single_cell_types, markers=False)
y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))

X_augmented_sum, X_augmented_sum_bin, X_augmented_mean, X_augmented_mean_bin,\
           X_augmented_max, X_augmented_max_bin, X_augmented_binary, y_nhot_augmented = \
    augment_data(X_single, y_single, n_celltypes, n_features, N_SAMPLES_PER_COMBINATION, label_encoder)

X_single = combine_samples(X_single)
X_single = X_single / 1000

X_mixtures, y_nhot_mixtures, mixture_label_encoder = \
    read_mixture_data(n_celltypes, label_encoder, binarize=False, markers=False)


def plot_distribution_original_datasets(savefig=None, show=None, singles=True):

    for single_cell_type in single_cell_types:
        single_cell_type_vec = string2vec([single_cell_type], label_encoder)
        if singles:
            indices_celltype = np.argwhere(np.max(np.multiply(y_nhot_single, single_cell_type_vec), axis=1) == 1).flatten().tolist()
        else:
            indices_celltype = np.argwhere(np.max(np.multiply(y_nhot_mixtures, single_cell_type_vec), axis=1) == 1).flatten().tolist()

        fig, axs = plt.subplots(3, 5)
        plt.suptitle(single_cell_type)
        for i in range(15):
            plt.subplot(3, 5, i+1)

            if singles:
                plt.hist(X_single[indices_celltype, i], density=True, histtype='bar')
            else:
                plt.hist(X_mixtures[indices_celltype, i], density=True, histtype='bar')
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            plt.title(present_markers[i], fontsize=10)

        single_cell_type = single_cell_type.replace(".", "_")
        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig+ '_{}'.format(single_cell_type))
            plt.close()
        if show or savefig is None:
            plt.show()

        plt.close(fig)


def plot_distribution_bin(savefig=None, show=None):

    for single_cell_type in single_cell_types:
        # get correct labels
        single_cell_type_vec = string2vec([single_cell_type], label_encoder)
        indices_celltype = np.argwhere(np.max(np.multiply(y_nhot_augmented, single_cell_type_vec), axis=1) == 1).flatten().tolist()

        fig, axs = plt.subplots(3, 5)
        plt.suptitle(single_cell_type)
        for i in range(15):
            plt.subplot(3, 5, i+1)
            colors = ['red', 'orange', 'green', 'blue']
            data = np.array([X_augmented_binary[indices_celltype, i],
                             X_augmented_sum_bin[indices_celltype, i],
                             X_augmented_mean_bin[indices_celltype, i],
                             X_augmented_max_bin[indices_celltype, i]])

            # delta_r_sum = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_sum_bin[indices_celltype, i] ** 2)
            # delta_r_mean = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_mean_bin[indices_celltype, i] ** 2)
            # delta_r_max = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_max_bin[indices_celltype, i] ** 2)
            #
            # deviation_sum = np.sum(delta_r_sum) / np.sum(X_augmented_binary[indices_celltype, i])
            # deviation_mean = np.sum(delta_r_mean) / np.sum(X_augmented_binary[indices_celltype, i])
            # deviation_max = np.sum(delta_r_max) / np.sum(X_augmented_binary[indices_celltype, i])

            # labels = ['binary', 'sum ; {}'.format(deviation_sum), 'mean ; {:1.2f}'.format(deviation_mean), 'max ; {}'.format(deviation_max)]

            labels = ['binary', 'sum', 'mean', 'max']

            plt.hist(data.T, density=True, histtype='bar', color=colors, label=labels)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            if i == 0:
                plt.legend(prop={'size': 6})
            plt.title(present_markers[i], fontsize=10)

        single_cell_type = single_cell_type.replace(".", "_")
        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_{}'.format(single_cell_type))
            plt.close()
        if show or savefig is None:
            plt.tight_layout()
            plt.show()

        plt.close(fig)


def plot_distribution_all(savefig=None, show=None):

    fig, axs = plt.subplots(3, 5)
    for i in range(15):
        plt.subplot(3, 5, i+1)

        colors = ['orange', 'green', 'blue']
        labels = ['sum', 'mean', 'max']
        data = np.array([X_augmented_sum[:, i],
                         X_augmented_mean[:, i],
                         X_augmented_max[:, i]])

        plt.hist(data.T, bins=20, histtype='bar', color=colors, label=labels, density=True)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.title(present_markers[i], fontsize=10)
        if i == 0:
            plt.legend(prop={'size': 6})

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()

    plt.close(fig)


def plot_distribution_ori(savefig=None, show=None):

    for single_cell_type in single_cell_types:
        # get correct labels
        single_cell_type_vec = string2vec([single_cell_type], label_encoder)
        indices_celltype = np.argwhere(np.max(np.multiply(y_nhot_augmented, single_cell_type_vec), axis=1) == 1).flatten().tolist()

        fig, axs = plt.subplots(3, 5)
        plt.suptitle(single_cell_type)
        for i in range(15):
            plt.subplot(3, 5, i+1)

            colors = ['orange', 'green', 'blue']
            labels = ['sum', 'mean', 'max']
            data = np.array([X_augmented_sum[indices_celltype, i],
                             X_augmented_mean[indices_celltype, i],
                             X_augmented_max[indices_celltype, i]])

            plt.hist(data.T, bins=20, histtype='bar', color=colors, label=labels)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title(present_markers[i], fontsize=10)
            if i == 0:
                plt.legend(prop={'size': 6})

        single_cell_type = single_cell_type.replace(".", "_")
        if savefig is not None:
            plt.tight_layout()
            plt.savefig(savefig + '_{}'.format(single_cell_type))
            plt.close()
        if show or savefig is None:
            plt.show()

        plt.close(fig)


if __name__ == '__main__':
    # plot_distribution_original_datasets(savefig=os.path.join('Plots', 'distribution_of_singles_dataset'))
    # plot_distribution_original_datasets(savefig=os.path.join('Plots', 'distribution_of_mixtures_dataset'), singles=False)
    plot_distribution_bin(savefig=os.path.join('Plots', 'distribution_of_binary'))
    # plot_distribution_all(savefig=os.path.join('Plots', 'distribution_of_all'))
    # plot_distribution_ori(savefig=os.path.join('Plots', 'distribution_of_ori'))



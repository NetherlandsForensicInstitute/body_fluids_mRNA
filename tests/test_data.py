"""
Compares augmented data based on taking the mean and on taking the sum.
"""

import numpy as np
import matplotlib.pyplot as plt

from rna.augment import MultiLabelEncoder, augment_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type
from rna.utils import string2vec

tc = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin',
      'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']
N_SAMPLES_PER_COMBINATION = 4
mle = MultiLabelEncoder(len(single_cell_types))

X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
    get_data_per_cell_type(single_cell_types=single_cell_types, markers=False)
y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
target_classes = string2vec(tc, label_encoder)

X_augmented_sum, X_augmented_max_binX_augmented_sum_bin, X_augmented_mean, X_augmented_mean_bin,\
           X_augmented_max, X_augmented_max_bin, X_augmented_binary, y_nhot_augmented = \
    augment_data(X_single, y_single, n_celltypes, n_features, N_SAMPLES_PER_COMBINATION, label_encoder, from_penile=False)

# fig, axs = plt.subplots(3, 5)
# for i in range(15):
#     plt.subplot(3, 5, i+1)
#     plt.hist(X_augmented_sum[:, i], label='sum', alpha=0.4)
#     plt.hist(X_augmented_mean[:, i], label='mean', alpha=0.4)
#     plt.hist(X_augmented_max[:, i], label='max', alpha=0.4)
#     plt.title(present_markers[i], fontsize=10)
#     if i == 0:
#         plt.legend()
#
# plt.tight_layout()
# plt.show()
# plt.close()

for single_cell_type in single_cell_types:
    print("\n", single_cell_type)
    print("------------------\n")
    # get correct labels
    single_cell_type_vec = string2vec([single_cell_type], label_encoder)
    indices_celltype = np.argwhere(np.max(np.multiply(y_nhot_augmented, single_cell_type_vec), axis=1) == 1).flatten().tolist()

    fig, axs = plt.subplots(3, 5)
    plt.suptitle(single_cell_type)
    for i in range(15):
        plt.subplot(3, 5, i+1)

        plt.hist(X_augmented_binary[indices_celltype, i], label='binary', color='orange', density=True)
        # plt.hist(X_augmented_sum_bin[indices_celltype, i], label='sum', alpha=0.5, color='blue', density=True)
        # plt.hist(X_augmented_mean_bin[indices_celltype, i], label='mean', alpha=0.5, color='blue', density=True)
        plt.hist(X_augmented_max_bin[indices_celltype, i], label='max', alpha=0.5, color='blue', density=True)
        plt.title(present_markers[i], fontsize=10)
        if i == 0:
            plt.legend()

        # delta_r = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_sum_bin[indices_celltype, i] ** 2)
        # delta_r = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_mean_bin[indices_celltype, i] ** 2)
        delta_r = np.sqrt(X_augmented_binary[indices_celltype, i] ** 2 - X_augmented_max_bin[indices_celltype, i] ** 2)
        deviation = np.sum(delta_r) / np.sum(X_augmented_binary[indices_celltype, i])
        print("Marker", present_markers[i], "deviates", round(deviation, 3))

    plt.tight_layout()
    plt.show()
    plt.close()


"""
Compares augmented data based on taking the mean and on taking the sum.
"""

import matplotlib.pyplot as plt

from rna.augment import MultiLabelEncoder, augment_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type
from rna.utils import string2vec

tc = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin',
      'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']
N_SAMPLES_PER_COMBINATION = 25
mle = MultiLabelEncoder(len(single_cell_types))

X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
    get_data_per_cell_type(single_cell_types=single_cell_types, markers=False)
y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
target_classes = string2vec(tc, label_encoder)

X_augmented_sum, X_augmented_sum_bin, X_augmented_mean, X_augmented_mean_bin, y_nhot_augmented = \
    augment_data(X_single, y_single, n_celltypes, n_features, N_SAMPLES_PER_COMBINATION, label_encoder, binarize=False, from_penile=False)

fig, axs = plt.subplots(3, 5)
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.hist(X_augmented_sum[:, i], label='sum', alpha=0.6)
    plt.hist(X_augmented_mean[:, i], label='mean', alpha=0.6)
    plt.title(present_markers[i], fontsize=10)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
plt.close()

fig, axs = plt.subplots(3, 5)
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.hist(X_augmented_sum_bin[:, i], label='sum', alpha=0.8)
    plt.hist(X_augmented_mean_bin[:, i], label='mean', alpha=0.4)
    plt.title(present_markers[i], fontsize=10)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
plt.close()


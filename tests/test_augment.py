import numpy as np

import rna.settings as settings

from rna.augment import augment_data, MultiLabelEncoder
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type
from rna.utils import string2vec


def test_augment_data(priors, y_nhot):
    """
    Tests that for given priors, the time that a cell type occurs
    is the same as the prior infers.
    """

    occurrence_celltypes = np.sum(y_nhot, axis=0)
    if len(np.unique(priors)) == 1 or priors is None:
        assert all(occurrence == occurrence_celltypes.tolist()[0] for occurrence in occurrence_celltypes.tolist())

    else:
        counts = {priors.count(value) : value for value in list(set(priors))}
        relevant_prior = counts[1]
        counts.pop(1)
        value_other_priors = list(counts.values())[0]

        index_of_relevant_prior = priors.index(relevant_prior)
        occurrence_of_relevant_prior = occurrence_celltypes[index_of_relevant_prior]

        relative_occurrence_of_relevant_celltype = float(occurrence_of_relevant_prior / y_nhot.shape[0])
        relative_occurrence_without_celltype = float((y_nhot.shape[0] - occurrence_of_relevant_prior) / y_nhot.shape[0])

        if relevant_prior != 1:
            assert round(relative_occurrence_of_relevant_celltype, 5) == \
                   round(relative_occurrence_without_celltype * relevant_prior, 5)
        else:
            assert round(relative_occurrence_of_relevant_celltype * value_other_priors, 5) == \
                   round(relative_occurrence_without_celltype, 5)


if __name__ == '__main__':
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))
    tc = ['Skin', 'Vaginal.mucosa and/or Menstrual.secretion']

    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    N_SAMPLES_PER_COMBINATION = [11, 22, 33]
    priors = [[1, 1, 1, 1, 1, 1, 1, 1],         # uniform priors
              [10, 1, 1, 1, 1, 1, 1, 1],        # cell type 1 occurs 10 times more often
              # [1, 1, 1, 7, 1, 1, 1, 1],         # cell type 4 occurs 7 times more often
              [1, 10, 10, 10, 10, 10, 10, 10],  # cell type 1 occurs 10 times less often
              [7, 7, 7, 7, 7, 1, 7, 7]]         # cell type 6 occurs 7 times less often

    for N_SAMPLES in N_SAMPLES_PER_COMBINATION:
        print(N_SAMPLES)
        for prior in priors:
            print(prior)
            X_augmented, y_nhot_augmented = augment_data(X_single, y_single, n_celltypes, n_features,
                                                                 N_SAMPLES, label_encoder, prior,
                                                                 binarize=settings.binarize, from_penile=from_penile)
            test_augment_data(prior, y_nhot_augmented)

    print("No assertion errors occurred.")
"""
Settings

Options:
    split_before                If provided, split the original data set before the nfold analysis starts, otherwise split the
                                original data set again at the start of each fold.
    augment                     If provided, use augmented data to train/calibrate the model with, otherwise use original data # TODO: make this work
    binarize                    If provided, make data binary, otherwise use the normalized original signal values
    markers                     If provided, include all markers, otherwise exclude 4 markers for control and gender
    softmax                     If provided, calculate the probabilities with softmax, otherwise use sigmoids
    nsamples                    The number of augmented samples per combination: (N_SAMPLES_TRAIN, N_SAMPLES_CALIB, N_SAMPLES_TEST)
                                Note that 22 = 4 and 11 = 2 # TODO: explain clearer
    test_size                   The size of the test data from the original NFI data set depending on total size of the data. The size of the train data = 1 - test_size.
    calibration_size            The size of the calibration data from the original NFI data set depending on the size of the residual train data.
                                If no separate data for calibration set to 0.0
                                An example: if test_size=0.2 and calibration_size=0.0, than the train_size=0.8. If calibration=0.5, than the actual
                                calibration_size=0.4 (and not 0.5!) and the actual train_size=0.4.
    calibration_on_loglrs       If provided, fit calibration model on 10loglrs, otherwise on the probabilities.
    from_penile                 If provided, always add penile skin in the mixtures created when augmenting data.
    models [model, bool]        Models is a list of lists [str, bool]. The model used for the analysis: 'MLR', 'MLP', 'XGB', 'DL'.
                                If boolean is True then perform with calibration otherwise no calibration.
                                An example: [['MLP', True], ['MLR', False], ['XGB', True], ['DL', True]] --> four models that are trained
                                and used to calculate LRs with. For 'MLP', 'XGB' and 'DL' calibration models are fitted and used to
                                transform the LRs (scores) into calibrated LRs.
    priors                      List of length 2 with vectors of length number of single cell types representing the prior distribution
                                of the augmented samples. [1, 1, 1, 1, 1, 1, 1, 1] are uniform priors. [10, 1, 1, 1, 1, 1, 1, 1] means
                                that samples with cell type at index 0 occurs 10 times more often than samples without that cell type.
                                Note that the first vector in the sample is considered the baseline distribution. So the augmented test
                                samples with that distribution will be the test data.
"""

# split_before=False
# augment=True
# binarize=[True, False]
# markers=False
# softmax=[True, False]
# nsamples=(33, 33, 22)
# test_size=0.2
# calibration_size=0.5
# calibration_on_loglrs=True
# from_penile=False # !only checked for 'MLR' and softmax=False whether from_penile=True works!
# models=[['MLR', False], ['MLP', True], ['XGB', True], ['DL', True]]
# priors=[[1, 1, 1, 1, 1, 1, 1, 1]]

split_before=False
augment=True
binarize=[True]
markers=False
softmax=[False]
nsamples=(33, 33, 22)
test_size=0.2
calibration_size=0.5
calibration_on_loglrs=True
from_penile=False
models=[['MLR', False]]
priors=[[1, 1, 1, 1, 1, 1, 1, 1]]
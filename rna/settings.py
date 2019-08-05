"""
Settings

Options:
    split_before                If provided, split the original data set before the nfold analysis starts
    augment                     If provided, use augmented data to train/calibrate the model with, otherwise use original data
    binarize                    If provided, make data binary, otherwise use original signal values
    markers                     If provided, include all markers, otherwise exclude 4 markers for control and gender
    softmax                     If provided, calculate the probabilities with softmax, otherwise use sigmoids
    nsamples                    The number of augmented samples per combination: (N_SAMPLES_TRAIN, N_SAMPLES_CALIB, N_SAMPLES_TEST)
                                Note that 22 = 4 and 11 = 2
    test_size                   The size of the test data depending on total size of the data. The size of the train data = 1 - test_size.
    calibration_size            The size of the calibration depending on the size of the residual train data.
                                If no separate data for calibration set to 0.0
    calibration_on_loglrs       If provided, fit calibration model on loglrs, otherwise on the probabilities.
    models [model, bool]        The model used for the analysis: 'MLR', 'MLP', 'XGB', 'DL'. If boolean is True then perform with calibration
                                otherwise no calibration.
    priors                      List of length 2 with vectors of length number of single cell types representing the prior distribution
                                of the augmented samples. [1, 1, 1, 1, 1, 1, 1, 1] are uniform priors. [10, 1, 1, 1, 1, 1, 1, 1] means
                                that samples with cell type at index 0 occurs 10 times more often than samples without that cell type.
                                Note that the first vector in the sample is considered the baseline distribution. So the augmented test
                                samples with that distribution will be the test data.
"""

split_before=False
augment=True
binarize=[True, False]
markers=False
softmax=[True, False]
nsamples=(44, 44, 22)
test_size=0.2
calibration_size=0.5
calibration_on_loglrs=True
models=[['MLP', True], ['MLR', False], ['XGB', True], ['DL', True]]
priors=[[1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1]]
"""
Settings

Options:
    augment           If provided, use augmented data to train/calibrate the model with, otherwise use original data
    binarize          If provided, make data binary, otherwise use original signal values
    markers           If provided, include all markers, otherwise exclude 4 markers for control and gender
    lps               If provided, use the label powerset method, otherwise use sigmoids
    cal_probs         If proved, calibrate models on probabilities, otherwise on log10LRs)
    nsamples          The number of augmented samples per combination: (N_SAMPLES_TRAIN, N_SAMPLES_CALIB, N_SAMPLES_TEST)
    test_size         The size of the test data depending on total size of the data. The size of the train data = 1 - test_size.
    calibration_size  The size of the calibration depending on the size of the residual train data.
                      If no separate data for calibration set to 0.0
    model, bool       The model used for the analysis: 'MLR', 'MLP', 'XGB'. If boolean is True then perform with calibration
                      otherwise no calibration.
    priors            List of length of number of single cell types
"""

augment=True
binarize=[True]
markers=False
softmax=[True]
nsamples=(5, 5, 2)
test_size=0.2
calibration_size=0.5
models=[['MLR', False], ['XGB', True]]
priors=[[1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1],
        None]
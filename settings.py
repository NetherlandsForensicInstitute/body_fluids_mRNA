"""
Settings

Options:
    augment           If provided, use augmented data to train/calibrate the model with, otherwise use original data
    binarize          If provided, make data binary, otherwise use original signal values
    markers           If provided, include all markers, otherwise exclude 4 markers for control and gender
    lps               If provided, use the label powerset method, otherwise use sigmoids
    cal_probs         If proved, calibrate models on probabilities, otherwise on log10LRs)
    nsamples          The number of augmented samples per combination: (N_SAMPLES_TRAIN, N_SAMPLES_CALIB, N_SAMPLES_TEST)
    test_size         The size of the test data
    calibration_size  The size of the calibration data. If no separate data set for calibration set to 0.0
    model             The model used for the analysis: 'MLR', 'MLP', 'XGB'
"""

augment=True
# binarize=True
markers=False
# lps=False
cal_probs=False # Not incorporated in single_analysis.py
nsamples=(50, 50, 25)
test_size=0.2
calibration_size=0.5
# model='MLR'
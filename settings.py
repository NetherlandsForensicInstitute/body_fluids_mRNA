"""
Settings that should be given in command line arguments.
Usage:
    run.py [--augment] [--binarize] [--markers] [--lps] [--nsamples] [--test_size] [--calibration_size] [--model]

Options:
    --augment           If provided, use augmented data to train/calibrate the model with, otherwise use original data
    --binarize          If provided, make data binary, otherwise use original signal values
    --markers           If provided, include all markers, otherwise exclude 4 markers for control and gender
    --lps               If provided, use the label powerset method, otherwise use sigmoids
    --nsamples          The number of augmented samples per combination: (N_SAMPLES, N_SAMPLES_TEST)
    --test_size         The size of the test data
    --calibration_size  The size of the calibration data. If no separate data set for calibration set to None
    --model             The model used for the analysis: ['MLR', 'MLP', 'DL', 'XGB']
"""

augment=True
binarize=True
markers=True
lps=True
nsamples=(4, 2)
test_size=0.2
calibration_size=0.4
model=[1, 0, 0, 0]
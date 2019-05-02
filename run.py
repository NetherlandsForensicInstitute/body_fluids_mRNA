"""
Note: Pipeline, currently not in use!
Usage:
    run.py [--augment] [--binarize] [--markers] [--lps] [--nsamples] [--test_size] [--calibration_size] [--model <str>]

Options:
    -h --help               Show this screen.
    --augment               If provided, use augmented data to train/calibrate the model with, otherwise use original data
    --binarize              If provided, make data binary, otherwise use original signal values
    --markers               If provided, include all markers, otherwise exclude 4 markers for control and gender
    --lps                   If provided, use the label powerset method, otherwise use sigmoids
    --nsamples              The number of augmented samples per combination: (N_SAMPLES, N_SAMPLES_TEST) [default: (100, 50)]
    --test_size             The size of the test data [default: 0.2]
    --calibration_size      The size of the calibration data. If no separate data set for calibration set to None [default: 0.4]
    --model <str>           Model to perform analysis with, either MLR, MLP, DL or XGB [default: 'MLP']
"""

from scratch.analysis import nfold_analysis
# TODO: Install docopt
# from docopt import docopt

if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']

    nfold_analysis(nfolds=1, tc=target_classes_str)





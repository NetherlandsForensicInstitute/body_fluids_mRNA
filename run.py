import random
import shutil
import warnings
import os

import rna.constants as constants
import numpy as np

from rna.analysis import nfold_analysis, makeplots, get_final_trained_mlr_model

"""
Settings

Options:
    split_before                If provided, split the original data set before the nfold analysis starts, otherwise split the
                                original data set again at the start of each fold.
    binarize                    If provided, make data binary, otherwise use the normalized original signal values
    remove_structural           If False, include all markers, otherwise exclude 4 markers for control and gender
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
                                Note that the first vector in the sample is considered the after_adjusting_dl distribution. So the augmented test
                                samples with that distribution will be the test data.
"""

params = {
'binarize_list':[False, True],

'remove_structural':True,

'softmax_list':[True, False],

'nsamples':(10, 10, 5),

'test_size':0.2,

'calibration_size':0.5,

'calibration_on_loglrs':True,

'from_penile':False, # !only checked for 'MLR' and softmax=False whether from_penile=True works!

'models_list':[
    ['MLR', True],
    # ['MLR', False],
    ['XGB', True],
    # ['DL', True],
    ['MLP', True],
    ['RF', True]
    ],


# NB the prior is currently used to adjust the number of samples of certain type in the training data. This system just looks at relative numbers
# it could/should also be used to encode the 0 and 1 options, as already exists but is not used in the augment_data function. For this, the values have to be between 0 and 1.
'priors_list':[
    [10, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
}




if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    nfolds=10

    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion']
    save_path = os.path.join('scratch','all_logreg_more_blood_seeded')

    shutil.rmtree(save_path,ignore_errors=True)
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, 'plots'))
    os.makedirs(os.path.join(save_path, 'picklesaves'))
    nfold_analysis(nfolds=nfolds, tc=target_classes_str, savepath=save_path, **params)

    makeplots(nfolds=nfolds, tc=target_classes_str, path=os.path.join(save_path, 'picklesaves'),
              savepath=save_path, **params)


    # save_path = 'final_model'
    # os.makedirs(save_path, exist_ok=True)
    # get_final_trained_mlr_model(tc=sorted(['Vaginal.mucosa and/or Menstrual.secretion'] + list(constants.single_cell_types)), retrain=True, n_samples_per_combination=10,
    #                             binarize=True, from_penile=False, prior=[10]+[1]*7, model_name=
    #                       constants.model_names['Vaginal mucosa and/or Menstrual secretion no Skin Penile'], save_path=save_path)


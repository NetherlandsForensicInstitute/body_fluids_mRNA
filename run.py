"""
Note: Pipeline, currently not in use!
Usage:
    run.py [--augment] [--markers] [--sigmoid] [--samples <s>] [--parameters <?>] [--test <?>]

Options:
    --augment           If provided, use augmented data to train/calibrate the model with
    --binarize          If provided, make data binary
    --markers           If provided, include all markers
    --sigmoid           If provided, estimates probability fo each individual cell type
    --samples <s>       The number of augmented samples per combination
    --model <?>         The model that the analysis is performed with
    --parameters <?>
    --test <?>          The type of data that is tested on
    --calibration <?>
"""

from scratch.analysis import nfold_analysis

if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types?
    from_penile = False
    retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']

    nfold_analysis(nfolds=1, tc=target_classes_str)





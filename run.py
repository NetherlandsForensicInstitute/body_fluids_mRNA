import time

from rna.nfold_analysis import nfold_analysis
from rna.test_priors import test_priors

if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Skin', 'Vaginal.mucosa and/or Menstrual.secretion']

    start = time.time()
    # nfold_analysis(nfolds=1, tc=target_classes_str)
    test_priors(nfolds=1, tc=target_classes_str)
    end = time.time()

    print("Execution time in seconds:", end - start)




import time

from rna.test_priors import test_priors

if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Nasal.mucosa', 'Saliva']

    start = time.time()
    test_priors(nfolds=1, tc=target_classes_str)
    end = time.time()

    print("Execution time in seconds:", end - start)
import time

from rna.nfold import nfold_analysis

if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion']

    start = time.time()
    nfold_analysis(nfolds=2, tc=target_classes_str)
    end = time.time()

    print("Execution time in seconds:", end - start)
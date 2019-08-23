import time
import warnings

from rna.nfold import nfold_analysis

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    # from_penile = False
    # retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Saliva', 'Skin', 'Nasal.mucosa', 'Menstrual.secretion', 'Blood']

    start = time.time()
    nfold_analysis(nfolds=100, tc=target_classes_str, savepath='scratch/analysisMLR/binsig100')
    end = time.time()

    print("Execution time in seconds:", end - start)
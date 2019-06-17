import time

from rna.nfold_analysis import nfold_analysis
from rna.single_analysis import single_analysis

## TEMPORARY
from rna.plotting import plot_distribution_of_samples
from rna.constants import single_cell_types


if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # TODO: finish making this plot
    # plot_distribution_of_samples(single_cell_types=single_cell_types, show=True)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']

    start = time.time()
    # nfold_analysis(nfolds=25, tc=target_classes_str)
    single_analysis(tc=target_classes_str)
    end = time.time()

    print("Execution time in seconds:", end - start)




import time
from scratch.nfold_analysis import nfold_analysis
from scratch.single_analysis import individual_analysis
import settings

## TEMPORARY
from rna.plotting import plot_distribution_of_samples
from rna.constants import single_cell_types


if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    plot_distribution_of_samples(single_cell_types=single_cell_types, show=True)

    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']

    start = time.time()
    # nfold_analysis(nfolds=10, tc=target_classes_str, settings.augment, settings.binarize, settings.markers, settings.nsamples,
    # settings.test_size, settings.calibration_size, settings.model)
    # nfold_analysis(nfolds=10, tc=target_classes_str)
    individual_analysis(tc=target_classes_str, treat_replicates_as_single=True)
    end = time.time()

    print("Execution time in seconds:", end - start)




import time

from rna.nfold_analysis import nfold_analysis


## TEMPORARY
import os
from rna.plotting import plot_distribution_of_samples, plot_distribution_of_mixture_samples, \
    plot_correlation_between_markers
from rna.constants import single_cell_types


if __name__ == '__main__':
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    from_penile = False
    retrain = True

    # TODO: finish making this plot
    plot_correlation_between_markers(single_cell_types=single_cell_types)
    # plot_distribution_of_samples(single_cell_types=single_cell_types, savefig=os.path.join('Plots', 'distribution_of_samples_singles_data'))
    # plot_distribution_of_samples(single_cell_types=single_cell_types)
    # plot_distribution_of_mixture_samples(savefig=os.path.join('Plots', 'distribution_of_samples_mixtures_data'))


    # assume that this is what comes from the GUI
    target_classes_str = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa', 'Vaginal.mucosa and/or Menstrual.secretion']

    start = time.time()
    nfold_analysis(nfolds=10, tc=target_classes_str)
    end = time.time()

    print("Execution time in seconds:", end - start)




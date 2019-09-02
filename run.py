import warnings

from rna.nfold import nfold_analysis

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    # from_penile = False
    # retrain = True

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Blood', 'Vaginal.mucosa',
                          'Menstrual.secretion', 'Saliva', 'Nasal.mucosa', 'Skin']

    nfold_analysis(nfolds=1, tc=target_classes_str, savepath='scratch/final_runs/baseline')
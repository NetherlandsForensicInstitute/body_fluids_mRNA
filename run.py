import warnings

from rna.nfold import nfold_analysis, makeplots

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    # TODO: boolean still needed? Implicitly taken into account with single_cell_types? --> Yes figure out how
    # from_penile = False
    # retrain = True
    nfolds=5
    run=1

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Blood', 'Vaginal.mucosa',
                          'Menstrual.secretion', 'Saliva', 'Nasal.mucosa', 'Skin']
    nfold_analysis(nfolds=nfolds, run=run, tc=target_classes_str, savepath='scratch/final_runs/baseline')

    makeplots(nfolds=nfolds, run=run, tc=target_classes_str, path='scratch/final_runs/baseline/picklesaves', savepath='scratch')
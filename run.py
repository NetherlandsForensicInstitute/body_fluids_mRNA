import warnings

from rna.nfold import nfold_analysis, makeplots

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    # retrain = True
    nfolds=1
    run=2

    # assume that this is what comes from the GUI
    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Blood', 'Vaginal.mucosa',
                          'Menstrual.secretion', 'Saliva', 'Nasal.mucosa', 'Skin']
    nfold_analysis(nfolds=nfolds, run=run, tc=target_classes_str, savepath='scratch')

    # makeplots(nfolds=nfolds, run=run, tc=target_classes_str, path='scratch/final_runs/baseline/picklesaves', savepath='scratch')
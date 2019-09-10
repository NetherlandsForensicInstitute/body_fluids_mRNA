import warnings

from rna.nfold import get_trained_mlr_model, nfold_analysis, makeplots

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    retrain=True
    nfolds=5
    run=7

    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion', 'Blood', 'Vaginal.mucosa',
                          'Menstrual.secretion', 'Saliva', 'Nasal.mucosa', 'Skin']
    # nfold_analysis(nfolds=nfolds, run=run, tc=target_classes_str, savepath='scratch/final_runs/baseline')
    makeplots(nfolds=36, run=run, tc=target_classes_str, path='scratch/final_runs/baseline/picklesaves', savepath='scratch')

    # get_trained_mlr_model(tc=['Vaginal.mucosa and/or Menstrual.secretion'], retrain=False, n_samples_per_combination=25,
    #                       binarize=True, from_penile=False, model_name='vagmenstr_no_penile')


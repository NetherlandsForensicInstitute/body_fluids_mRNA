import warnings
import rna.constants as constants

from rna.analysis import get_trained_mlr_model, nfold_analysis, makeplots

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    nfolds=20
    run=1

    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion']
    # nfold_analysis(nfolds=nfolds, run=run, tc=target_classes_str, savepath='scratch/final_runs/only_vag_menstr')
    makeplots(nfolds=25, run=run, tc=target_classes_str, path='scratch/final_runs/only_vag_menstr/picklesaves',
              savepath='scratch/final_runs/only_vag_menstr')

    # get_trained_mlr_model(tc=['Vaginal.mucosa and/or Menstrual.secretion'], retrain=False, n_samples_per_combination=50,
    #                       binarize=True, from_penile=False, model_name=
    #                       constants.model_names['Vaginal mucosa and/or Menstrual secretion no Skin Penile'])
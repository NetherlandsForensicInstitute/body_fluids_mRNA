import warnings
import rna.constants as constants

from rna.analysis import get_trained_mlr_model, nfold_analysis, makeplots

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # to ignore RuntimeError: divide by zero.
    nfolds=200
    run=1

    target_classes_str = ['Vaginal.mucosa and/or Menstrual.secretion']
    save_path = 'scratch/only_vag_menstr_5_5_5'
    nfold_analysis(nfolds=nfolds, run=run, tc=target_classes_str, savepath=save_path)
    makeplots(nfolds=nfolds, run=run, tc=target_classes_str, path=save_path+'/picklesaves',
              savepath=save_path)

    # get_trained_mlr_model(tc=['Vaginal.mucosa and/or Menstrual.secretion'], retrain=False, n_samples_per_combination=50,
    #                       binarize=True, from_penile=False, model_name=
    #                       constants.model_names['Vaginal mucosa and/or Menstrual secretion no Skin Penile'])
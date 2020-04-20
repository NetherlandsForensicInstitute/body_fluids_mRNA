"""

"""

import os
import pickle
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, append_lrs_for_all_folds, clf_with_correct_settings
from rna.augment import MultiLabelEncoder, augment_splitted_data, binarize_and_combine_samples
from rna.constants import single_cell_types, marker_names
from rna.input_output import get_data_per_cell_type, read_mixture_data, \
    save_data_table
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax, LrsBeforeAfterCalib
from rna.plotting import plot_scatterplots_all_lrs_different_priors, plot_boxplot_of_metric, \
    plot_progress_of_metric, plot_coefficient_importances, plot_histograms_all_lrs_all_folds
from rna.lr_system import MarginalClassifier


def get_final_trained_mlr_model(tc, retrain, n_samples_per_combination, binarize=True, from_penile=False, prior=(1,1,1,1,1,1,1,1), model_name='best_MLR', remove_structural=True, save_path=None):
    """
    computes or loads the MLR based on all data
    """
    mle = MultiLabelEncoder(len(single_cell_types))

    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=True)
    
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    X_all, y_all, _, _, _, \
    label_encoder_all, present_markers, _ = \
        get_data_per_cell_type(single_cell_types= ('Blood', 'Saliva', 'Vaginal.mucosa', 'Menstrual.secretion',
     'Semen.fertile', 'Semen.sterile', 'Nasal.mucosa', 'Skin', 'Skin.penile'),
                               remove_structural=True)

    save_data_table(X_all, [vec2string(y, label_encoder_all) for
                               y in y_all],
                    present_markers, os.path.join(save_path, 'single cell '
                                                             'data.csv'))
    if retrain:
        model = clf_with_correct_settings('MLR', softmax=False, n_classes=-1, with_calibration=True)
        X_train, X_calib, y_train, y_calib = train_test_split(X_single, y_single, stratify=y_single, test_size=0.5)

        X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, remove_structural=remove_structural)

        save_data_table(X_mixtures, [vec2string(y, label_encoder).replace(' and/or ', '+') for y in
                                     y_nhot_mixtures],
                        present_markers, os.path.join(save_path, 'mixture '
                                                             'data.csv'))

        augmented_data = augment_splitted_data(X_train, y_train, X_calib, y_calib, None, None,
                                                            y_nhot_mixtures, n_celltypes, n_features,
                                                            label_encoder, prior, [binarize],
                                                            from_penile, [n_samples_per_combination]*3)

        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(augmented_data.y_train_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T
        y_calib = np.array([np.max(np.array(augmented_data.y_calib_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T

        model.fit_classifier(augmented_data.X_train_augmented, y_train)
        model.fit_calibration(augmented_data.X_calib_augmented, augmented_data.y_calib_nhot_augmented, target_classes)
        pickle.dump(model, open('{}'.format(model_name), 'wb'))
    else:
        model = pickle.load(open('{}'.format(model_name), 'rb'))


    compare_to_multiclass(model, np.array([[1,1,1,0,0,0,1,1,1,1,1,1,0,0,0], [0]*4+[1,1]+[0]*9, [0]*3+[1, 1,1, 1]+[0]*8]), remove_structural=remove_structural, tc=tc, save_path=save_path)

    # plot the coefficients
    plot_coefficient_importances(model, target_classes, present_markers, label_encoder, savefig=os.path.join(save_path, 'coefs_{}_{}'.format(prior, model_name)), show=None)


    t = np.argwhere(np.array(tc)=='Vaginal.mucosa and/or Menstrual.secretion').squeeze()
    intercept, coefficients = model.get_coefficients(t, target_classes[t].squeeze())
    all_coefficients = np.append(intercept, coefficients).tolist()
    all_coefficients_str = [str(coef) for coef in all_coefficients]
    all_coefficients_strr = [coef.replace('.', ',') for coef in all_coefficients_str]
    present_markers.insert(0, 'intercept')

    with open(os.path.join(save_path,'coefs_{}_{}.csv'.format(prior, model_name)), mode='w') as coefs:
        coefs_writer = csv.writer(coefs, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        coefs_writer.writerow(present_markers)
        coefs_writer.writerow(all_coefficients_strr)


def nfold_analysis(nfolds, tc, savepath, from_penile: bool, models_list, softmax_list: List[bool], priors_list: List[List], binarize_list: List[bool], test_size: float, calibration_size: float, remove_structural: bool, calibration_on_loglrs: bool, nsamples: Tuple[int, int, int]):
    # if from_penile == True:
    #     if True in softmax_list:
    #         raise ValueError("The results following from these settings have not been validated and hence cannot be "
    #                          "relied on. Make sure 'softmax' is set to False if 'from_penile' is {}".format(
    #             from_penile))
    #     for models_and_calib in models_list:
    #         if 'MLP' in models_and_calib or 'XGB' in models_and_calib or 'DL' in models_and_calib:
    #             raise ValueError("The results following from these settings have not validated and hence cannot be "
    #                              "relied on. The model cannot be {} if 'from_penile' is {}. Either adjust the model "
    #                              "to 'MLR' or set 'from_penile=False'.".format(models_and_calib[0], from_penile))

    mle = MultiLabelEncoder(len(single_cell_types))
    baseline_prior = str(priors_list[0])

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=remove_structural)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)



    outer = tqdm(total=nfolds, desc='{} folds'.format(nfolds), position=0, leave=False)
    for n in range(nfolds):
        # n = n + (nfolds * run)
        print(n)

        # ======= Initialize =======
        lrs_for_model_in_fold = OrderedDict()
        emtpy_numpy_array = np.zeros((len(binarize_list), len(softmax_list), len(models_list), len(priors_list)))
        accuracies_train_n, accuracies_test_n, accuracies_test_as_mixtures_n, accuracies_mixtures_n, accuracies_single_n,\
        cllr_test_n, cllr_test_as_mixtures_n, cllr_mixtures_n, coeffs = [dict() for i in range(9)]

        for target_class in target_classes:
            target_class_str = vec2string(target_class, label_encoder)

            accuracies_train_n[target_class_str] = emtpy_numpy_array.copy()
            accuracies_test_n[target_class_str] = emtpy_numpy_array.copy()
            accuracies_test_as_mixtures_n[target_class_str] = emtpy_numpy_array.copy()
            accuracies_mixtures_n[target_class_str] = emtpy_numpy_array.copy()
            accuracies_single_n[target_class_str] = emtpy_numpy_array.copy()

            cllr_test_n[target_class_str] = emtpy_numpy_array.copy()
            cllr_test_as_mixtures_n[target_class_str] = emtpy_numpy_array.copy()
            cllr_mixtures_n[target_class_str] = emtpy_numpy_array.copy()
            coeffs[target_class_str] = np.zeros((len(binarize_list),1,X_single[0].shape[1]+1, len(priors_list)))
        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=calibration_size)

        for i, binarize in enumerate(binarize_list):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, remove_structural=remove_structural)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(priors_list):
                augmented_data[str(priors)] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                    y_nhot_mixtures, n_celltypes, n_features,
                                                                    label_encoder, priors, binarize_list,
                                                                    from_penile, nsamples)

            # ======= Transform data accordingly =======
            if binarize:
                X_test_transformed = [
                    [np.where(X_test[i][j] > 150, 1, 0) for j in range(len(X_test[i]))] for i in
                    range(len(X_test))]
                X_test_transformed = combine_samples(X_test_transformed)
            else:
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(softmax_list):
                for k, model_calib in enumerate(models_list):
                    print(model_calib[0])


                    # ======= Calculate LRs before and after calibration =======
                    key_name = bool2str_binarize(binarize) + '_' + bool2str_softmax(softmax) + '_' + str(model_calib)
                    if not model_calib[1]:
                        key_name+='_uncal'
                    key_name_per_fold = str(n) + '_' + key_name
                    model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                    lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
                    lrs_before_calib_mixt, lrs_after_calib_mixt = \
                        calculate_lrs_for_different_priors(augmented_data, X_mixtures, target_classes, baseline_prior,
                                                           present_markers, model_calib, mle, label_encoder, key_name_per_fold,
                                                           softmax, calibration_on_loglrs, savepath)

                    lrs_for_model_in_fold[key_name] = LrsBeforeAfterCalib(lrs_before_calib, lrs_after_calib, y_test_nhot_augmented,
                                                                          lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented,
                                                                          lrs_before_calib_mixt, lrs_after_calib_mixt, y_nhot_mixtures)

                    ## Check which samples the method makes an error with
                    # indices_values_above_one = np.argwhere(lrs_for_model_in_fold['[1, 1, 1, 1, 1, 1, 1, 1]'].lrs_before_calib > 1)[:, 0]
                    # indices_values_below_one = np.argwhere(lrs_for_model_in_fold['[1, 1, 1, 1, 1, 1, 1, 1]'].lrs_before_calib < 1)[:, 0]
                    # labels = np.max(np.multiply(augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented, target_class), axis=1)
                    # indices_fp = np.argwhere(labels[indices_values_above_one] == 0)
                    # indices_fn = np.argwhere(labels[indices_values_below_one] == 1)
                    # augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented[indices_values_above_one][indices_fp][:, 0, :]
                    # augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented[indices_values_below_one][indices_fn][:, 0, :]


                    # ======= Calculate performance metrics =======
                    for t, target_class in enumerate(target_classes):
                        for p, priors in enumerate(priors_list):
                            str_prior = str(priors)
                            target_class_str = vec2string(target_class, label_encoder)

                            accuracies_train_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[str_prior].X_train_augmented,
                                augmented_data[str_prior].y_train_nhot_augmented, target_classes, model[str_prior],
                                mle)[t]
                            accuracies_test_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_augmented,
                                augmented_data[baseline_prior].y_test_nhot_augmented, target_classes, model[str_prior],
                                mle)[t]
                            accuracies_test_as_mixtures_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_as_mixtures_augmented,
                                augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_classes,
                                model[str_prior], mle)[t]
                            accuracies_mixtures_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_mixtures, y_nhot_mixtures, target_classes, model[str_prior], mle)[t]
                            accuracies_single_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_test_transformed, mle.inv_transform_single(y_test), target_classes, model[str_prior],
                                mle)[t]

                            cllr_test_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib[str_prior][:, t], augmented_data[baseline_prior].y_test_nhot_augmented, target_class)
                            cllr_test_as_mixtures_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib_test_as_mixtures[str_prior][:, t], augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_class)
                            cllr_mixtures_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib_mixt[str_prior][:, t], y_nhot_mixtures, target_class)
                            if model_calib[0] == 'MLR' and not softmax:
                                # save coefficents
                                intercept, coefficients = model[str(priors)].get_coefficients(t, target_class)
                                coeffs[target_class_str][i, 0, 0, p] = intercept
                                for i_coef, coef in enumerate(coefficients):
                                    coeffs[target_class_str][i, 0, i_coef+1, p] = coef

        outer.update(1)


        # ======= Save lrs and performance metrics =======
        pickle.dump(lrs_for_model_in_fold, open(os.path.join(savepath, 'picklesaves/lrs_for_model_in_fold_{}'.format(n)), 'wb'))

        for t, target_class in enumerate(target_classes):
            target_class_str = vec2string(target_class, label_encoder)
            target_class_save = target_class_str.replace(" ", "_")
            target_class_save = target_class_save.replace(".", "_")
            target_class_save = target_class_save.replace("/", "_")

            pickle.dump(accuracies_train_n[target_class_str], open(os.path.join(savepath, 'picklesaves/accuracies_train_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(accuracies_test_n[target_class_str], open(os.path.join(savepath, 'picklesaves/accuracies_test_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(accuracies_test_as_mixtures_n[target_class_str], open(os.path.join(savepath, 'picklesaves/accuracies_test_as_mixt_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(accuracies_mixtures_n[target_class_str], open(os.path.join(savepath, 'picklesaves/accuracies_mixt_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(accuracies_single_n[target_class_str], open(os.path.join(savepath, 'picklesaves/accuracies_single_{}_{}'.format(target_class_save, n)), 'wb'))

            pickle.dump(cllr_test_n[target_class_str], open(os.path.join(savepath, 'picklesaves/cllr_test_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(cllr_test_as_mixtures_n[target_class_str], open(os.path.join(savepath, 'picklesaves/cllr_test_as_mixt_{}_{}'.format(target_class_save, n)), 'wb'))
            pickle.dump(cllr_mixtures_n[target_class_str], open(os.path.join(savepath, 'picklesaves/cllr_mixt_{}_{}'.format(target_class_save, n)), 'wb'))

            pickle.dump(coeffs[target_class_str], open(os.path.join(savepath, 'picklesaves/coeffs_{}_{}'.format(target_class_save, n)), 'wb'))


def compare_to_multiclass(model: MarginalClassifier, samples, tc, remove_structural=True, binarize=True, save_path=None):
    """
    trains a multiclass model and compares its output to the given multilabel model, on a certain sample
    """

    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=remove_structural)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    X = binarize_and_combine_samples(X_single, binarize)

    multi_class_model = LogisticRegression()
    multi_class_model.fit(X, y_single)
    for sample in samples:
        sample=sample.reshape(1, -1)
        multi_pred = multi_class_model.predict_proba(sample)[0]
        prob_vag_menst = multi_pred[1]+multi_pred[7]
        multi_log_lrs = np.log10(np.append(multi_pred/(1-multi_pred), prob_vag_menst/(1-prob_vag_menst)))
        # take single cell target classes
        log_lrs = np.log10(model.predict_lrs(sample, target_classes))
        plt.figure()
        df=pd.DataFrame({'multiclass log(LR)': multi_log_lrs, 'multi-label log(LR)': log_lrs[0]})
        p =  sns.scatterplot(data=df, x='multiclass log(LR)', y= 'multi-label log(LR)')

        # add annotations one by one with a loop
        for line in range(0, df.shape[0]):
            p.text(multi_log_lrs[line] + 0.2, log_lrs[0][line], tc[line], horizontalalignment='left', size='medium', color='black',
                    weight='semibold')

        plt.savefig(os.path.join(save_path, 'loglrs_for_' + str(sample)))


def makeplots(tc, path, savepath, remove_structural: bool, nfolds, binarize_list, softmax_list, models_list, priors_list, **kwargs):

    _, _, _, _, _, label_encoder, _, _ = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=remove_structural)
    target_classes = string2vec(tc, label_encoder)

    lrs_for_model_per_fold = OrderedDict()
    emtpy_numpy_array = np.zeros(
        (nfolds, len(binarize_list), len(softmax_list), len(models_list), len(priors_list)))
    accuracies_train, accuracies_test, accuracies_test_as_mixtures, accuracies_mixtures, accuracies_single, \
    cllr_test, cllr_test_as_mixtures, cllr_mixtures, coeffs = [dict() for i in range(9)]

    for target_class in target_classes:
        target_class_str = vec2string(target_class, label_encoder)

        accuracies_train[target_class_str] = emtpy_numpy_array.copy()
        accuracies_test[target_class_str] = emtpy_numpy_array.copy()
        accuracies_test_as_mixtures[target_class_str] = emtpy_numpy_array.copy()
        accuracies_mixtures[target_class_str] = emtpy_numpy_array.copy()
        accuracies_single[target_class_str] = emtpy_numpy_array.copy()

        cllr_test[target_class_str] = emtpy_numpy_array.copy()
        cllr_test_as_mixtures[target_class_str] = emtpy_numpy_array.copy()
        cllr_mixtures[target_class_str] = emtpy_numpy_array.copy()
        coeffs[target_class_str] = np.zeros((nfolds, len(binarize_list),1, len(marker_names)-4+1, len(priors_list)))

    for n in range(nfolds):
        lrs_for_model_per_fold[str(n)] = pickle.load(open(os.path.join(path, 'lrs_for_model_in_fold_{}'.format(n)), 'rb'))

        for target_class in target_classes:
            target_class_str = vec2string(target_class, label_encoder)
            target_class_save = target_class_str.replace(" ", "_")
            target_class_save = target_class_save.replace(".", "_")
            target_class_save = target_class_save.replace("/", "_")

            accuracies_train[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path, 'accuracies_train_{}_{}'.format(target_class_save, n)), 'rb'))
            accuracies_test[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path,'accuracies_test_{}_{}'.format(target_class_save, n)), 'rb'))
            accuracies_test_as_mixtures[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path,'accuracies_test_as_mixt_{}_{}'.format(target_class_save, n)), 'rb'))
            accuracies_mixtures[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path,'accuracies_mixt_{}_{}'.format(target_class_save, n)), 'rb'))
            accuracies_single[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path, 'accuracies_single_{}_{}'.format(target_class_save, n)), 'rb'))

            cllr_test[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path, 'cllr_test_{}_{}'.format(target_class_save, n)), 'rb'))
            cllr_test_as_mixtures[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path, 'cllr_test_as_mixt_{}_{}'.format(target_class_save, n)), 'rb'))
            cllr_mixtures[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path,'cllr_mixt_{}_{}'.format(target_class_save, n)), 'rb'))
            coeffs[target_class_str][n, :, :, :, :] = pickle.load(
                open(os.path.join(path,'coeffs_{}_{}'.format(target_class_save, n)), 'rb'))

    types_data = ['test augm', 'mixt']  # 'mixt' and/or 'test augm as mixt'

    for type_data in types_data:
        lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = append_lrs_for_all_folds(
            lrs_for_model_per_fold, type=type_data)

        # plot_pavs_all_methods(lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods,
        #                           target_classes, label_encoder, savefig=os.path.join(savepath, 'pav_{}'.format(type_data)))
        #
        # # plot_rocs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder)
        #           # savefig=os.path.join(savepath, 'roc_{}'.format(type_data)))
        #
        plot_histograms_all_lrs_all_folds(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes,
                                          label_encoder,
                                          savefig=os.path.join(savepath, 'histograms_after_calib_{}'.format(type_data)))

        if len(priors_list) == 2:
            plot_scatterplots_all_lrs_different_priors(lrs_after_for_all_methods, y_nhot_for_all_methods,
                                                       target_classes, label_encoder,
                                                       savefig=os.path.join(savepath,
                                                                            'LRs_for_different_priors_{}'.format(
                                                                                type_data)))
    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, priors_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                               savefig=os.path.join(savepath, 'boxplot_accuracy_train_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, priors_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_train_{}'.format(target_class_save)))

        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, priors_list, accuracies_test[target_class_str], label_encoder, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_test_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, priors_list, accuracies_test[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, priors_list, cllr_test[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, priors_list, cllr_test[target_class_str], label_encoder, '$C_{llr}$',
                                savefig=os.path.join(savepath, 'progress_cllr_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, priors_list, cllr_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, priors_list, cllr_test_as_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_as_mixt_{}'.format(target_class_save)))

        plot_boxplot_of_metric(binarize_list, [False], [[a, True] for a in ['intercept']+marker_names], priors_list, coeffs[target_class_str], label_encoder, "log LR",
                               savefig=os.path.join(savepath, 'boxplot_coefficients_{}'.format(target_class_save)), ylim=[-3,3])


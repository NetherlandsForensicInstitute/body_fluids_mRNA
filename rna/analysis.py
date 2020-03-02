"""

"""

import csv
import os
import pickle
from collections import OrderedDict
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, append_lrs_for_all_folds, clf_with_correct_settings
from rna.augment import MultiLabelEncoder, augment_splitted_data, binarize_and_combine_samples
from rna.constants import marker_names
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import MarginalClassifier
from rna.plotting import plot_scatterplots_all_lrs_different_priors, plot_boxplot_of_metric, \
    plot_progress_of_metric, plot_coefficient_importances
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax, LrsBeforeAfterCalib, \
    priors_dict_to_list


def get_final_trained_mlr_model(tc, retrain, n_samples, binarize, priors_dict, model_name='best_MLR', remove_structural=True, save_path=None):
    """
    computes or loads the MLR based on all data
    """
    single_cell_types = list(priors_dict.keys())
    mle = MultiLabelEncoder(len(single_cell_types))


    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=True)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    baseline_prior, priors_list = priors_dict_to_list(label_encoder, n_celltypes, priors_dict)

    if retrain:
        model = clf_with_correct_settings('MLR', softmax=False, n_classes=-1, with_calibration=True)
        X_train, X_calib, y_train, y_calib = train_test_split(X_single, y_single, stratify=y_single, test_size=0.5)

        X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, remove_structural=remove_structural)

        augmented_data = augment_splitted_data(X_train, y_train, X_calib, y_calib, None, None,
                                               y_nhot_mixtures, n_celltypes, n_features,
                                               label_encoder, baseline_prior, [binarize],
                                               [n_samples] * 3)

        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(augmented_data.y_train_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T
        y_calib = np.array([np.max(np.array(augmented_data.y_calib_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T

        model.fit_classifier(augmented_data.X_train_augmented, y_train)
        model.fit_calibration(augmented_data.X_calib_augmented, augmented_data.y_calib_nhot_augmented, target_classes)
        pickle.dump(model, open('{}'.format(model_name), 'wb'))
    else:
        model = pickle.load(open('{}'.format(model_name), 'rb'))

    compare_to_multiclass(model, np.array([[1,1,1,0,0,0,1,1,1,1,1,1,0,0,0], [0]*4+[1,1]+[0]*9, [0]*3+[1, 1,1, 1]+[0]*8]), single_cell_types=single_cell_types, remove_structural=remove_structural, tc=tc, save_path=save_path)

    # plot the coefficients
    plot_coefficient_importances(model, target_classes, present_markers, label_encoder, savefig=os.path.join(save_path, 'coefs_{}_{}'.format(baseline_prior, model_name)), show=None)


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


def nfold_analysis(nfolds, tc, savepath, models_list, softmax_list: List[bool], priors_dict: Dict[str, List], binarize_list: List[bool], test_size: float, calibration_size: float, remove_structural: bool, calibration_on_loglrs: bool, nsamples: Tuple[int, int, int]):
    single_cell_types = list(priors_dict.keys())
    mle = MultiLabelEncoder(len(single_cell_types))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=remove_structural)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    baseline_prior, priors_list = priors_dict_to_list(label_encoder, n_celltypes, priors_dict)


    outer = tqdm(total=nfolds, desc='{} folds'.format(nfolds), position=0, leave=False)
    for n in range(nfolds):
        print(n)

        # ======= Initialize =======
        lrs_for_model_in_fold = OrderedDict()
        emtpy_numpy_array = np.zeros((len(binarize_list), len(softmax_list), len(models_list), len(priors_list)))
        accuracies_train_n, accuracies_test_n, accuracies_test_as_mixtures_n, accuracies_mixtures_n, accuracies_single_n, \
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
                augmented_data[priors] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                    y_nhot_mixtures, n_celltypes, n_features,
                                                                    label_encoder, priors, binarize_list,
                                                                    nsamples)

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

                    # ======= Calculate performance metrics =======
                    for t, target_class in enumerate(target_classes):
                        for p, priors in enumerate(priors_list):
                            target_class_str = vec2string(target_class, label_encoder)

                            accuracies_train_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[priors].X_train_augmented,
                                augmented_data[priors].y_train_nhot_augmented, target_classes, model[priors],
                                mle)[t]
                            accuracies_test_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_augmented,
                                augmented_data[baseline_prior].y_test_nhot_augmented, target_classes, model[priors],
                                mle)[t]
                            accuracies_test_as_mixtures_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_as_mixtures_augmented,
                                augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_classes,
                                model[priors], mle)[t]
                            accuracies_mixtures_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_mixtures, y_nhot_mixtures, target_classes, model[priors], mle)[t]
                            accuracies_single_n[target_class_str][i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_test_transformed, mle.inv_transform_single(y_test), target_classes, model[priors],
                                mle)[t]

                            cllr_test_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib[priors][:, t], augmented_data[baseline_prior].y_test_nhot_augmented, target_class)
                            cllr_test_as_mixtures_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib_test_as_mixtures[priors][:, t], augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_class)
                            cllr_mixtures_n[target_class_str][i, j, k, p] = cllr(
                                lrs_after_calib_mixt[priors][:, t], y_nhot_mixtures, target_class)
                            if model_calib[0] == 'MLR' and not softmax:
                                # save coefficents
                                intercept, coefficients = model[priors].get_coefficients(t, target_class)
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


def compare_to_multiclass(model: MarginalClassifier, samples, tc, single_cell_types, remove_structural=True, binarize=True, save_path=None):
    """
    trains a multiclass model and compares its output to the given multilabel model, on a certain sample
    """
    single_cell_types_to_use = set(tc).intersection(set(single_cell_types))
    mle = MultiLabelEncoder(len(single_cell_types_to_use))

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types_to_use, remove_structural=remove_structural)
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
        plt.rc('text', usetex=False)

        p =  sns.scatterplot(data=df, x='multiclass log(LR)', y= 'multi-label log(LR)')

        # add annotations one by one with a loop
        for line in range(0, df.shape[0]):
            p.text(multi_log_lrs[line] + 0.2, log_lrs[0][line], tc[line], horizontalalignment='left', size='medium', color='black',
                    weight='semibold')

        plt.savefig(os.path.join(save_path, 'loglrs_for_' + str(sample)))


def makeplots(tc, path, savepath, remove_structural: bool, nfolds, binarize_list, softmax_list, models_list, priors_dict, **kwargs):

    _, _, _, _, _, label_encoder, _, _ = \
        get_data_per_cell_type(single_cell_types=list(priors_dict.keys()), remove_structural=remove_structural)
    target_classes = string2vec(tc, label_encoder)

    n_priors = len(priors_dict['Blood'])
    lrs_for_model_per_fold = OrderedDict()
    emtpy_numpy_array = np.zeros(
        (nfolds, len(binarize_list), len(softmax_list), len(models_list), n_priors))
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
        coeffs[target_class_str] = np.zeros((nfolds, len(binarize_list),1, len(marker_names)-4+1, n_priors))

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
        # plot_histograms_all_lrs_all_folds(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes,
        #                                   label_encoder,
        #                                   savefig=os.path.join(savepath, 'histograms_after_calib_{}'.format(type_data)))

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

        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                               savefig=os.path.join(savepath, 'boxplot_accuracy_train_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_train_{}'.format(target_class_save)))

        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, accuracies_test[target_class_str], label_encoder, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_test_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, accuracies_test[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_test[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_progress_of_metric(binarize_list, softmax_list, models_list, cllr_test[target_class_str], label_encoder, '$C_{llr}$',
                                savefig=os.path.join(savepath, 'progress_cllr_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_test_as_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_as_mixt_{}'.format(target_class_save)))

        plot_boxplot_of_metric(binarize_list, [False], [[a, True] for a in ['intercept']+marker_names], coeffs[target_class_str], label_encoder, "log LR",
                               savefig=os.path.join(savepath, 'boxplot_coefficients_{}'.format(target_class_save)), ylim=[-3,3])


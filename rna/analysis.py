"""

"""

import os
import pickle
import csv

import numpy as np

from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from rna import constants
from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, append_lrs_for_all_folds, clf_with_correct_settings
from rna.augment import MultiLabelEncoder, augment_splitted_data, binarize_and_combine_samples
from rna.constants import single_cell_types, marker_names, DEBUG
from rna.input_output import get_data_per_cell_type, read_mixture_data, \
    save_data_table
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax, LrsBeforeAfterCalib
from rna.plotting import plot_scatterplots_all_lrs_different_priors, plot_boxplot_of_metric, \
    plot_progress_of_metric, plot_coefficient_importances, plot_property_all_lrs_all_folds, plot_multiclass_comparison
from rna.lr_system import MarginalClassifier


def get_final_trained_mlr_model(tc, retrain,
                                n_samples_per_combination,
                                binarize=True,
                                priors1=[],
                                priors0=[],
                                model_name='best_MLR',
                                remove_structural=True, save_path=None,
                                alternative_hypothesis=None,
                                # blood, nasal, vaginal
                                samples_to_evaluate=np.array([[1] * 3 + [0] + [1] * 5 + [0] * 6]),
                                use_mixtures=True):

    """
    computes or loads the MLR based on all data
    """
    softmax = False
    mle = MultiLabelEncoder(len(constants.single_cell_types))

    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=constants.single_cell_types, remove_structural=True)

    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    save_data_table(X_single, [vec2string(y, label_encoder) for
                               y in y_nhot_single],
                    present_markers, os.path.join(save_path, 'single cell data.csv'))

    if retrain:
        model = clf_with_correct_settings('MLR', softmax=softmax, n_classes=-1, with_calibration=True)
        X_train, X_calib, y_train, y_calib = train_test_split(X_single, y_single, stratify=y_single, test_size=0.5)
        if use_mixtures:
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder,
                                                                                   binarize=binarize,
                                                                                   remove_structural=remove_structural)

            save_data_table(
                X_mixtures,
                [vec2string(y, label_encoder).replace(' and/or ', '+') for y in
                                         y_nhot_mixtures],
                present_markers,
                os.path.join(save_path, 'mixture data.csv'))
        augmented_data = augment_splitted_data(X_train, y_train, X_calib, y_calib, None, None,
                                                            None, n_celltypes, n_features,
                                                            label_encoder, priors0, priors1, [binarize],
                                                            [n_samples_per_combination]*3,
                                                            disallowed_mixtures=None)

        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(augmented_data.y_train_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T
        # y_calib = np.array([np.max(np.array(augmented_data.y_calib_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T

        model.fit_classifier(augmented_data.X_train_augmented, y_train)
        model.fit_calibration(augmented_data.X_calib_augmented, augmented_data.y_calib_nhot_augmented, target_classes)
        pickle.dump(model, open('{}'.format(os.path.join(save_path,
                                                         model_name)), 'wb'))
    else:
        model = pickle.load(open('{}'.format(os.path.join(save_path,
                                                         model_name)), 'rb'))

    if alternative_hypothesis:
        # also plot LRs of our hypothesis pairs against LRs when H2 is more specific
        implied_target = string2vec(['Vaginal.mucosa and/or Menstrual.secretion'], label_encoder)
        alternative_target = string2vec(alternative_hypothesis, label_encoder)
        # at least one of H1 or alternative should be present, disallow absence of all:
        disallowed_mixtures = (-implied_target - alternative_target).astype(np.int)

        X_train, X_calib, y_train, y_calib = train_test_split(X_single, y_single, stratify=y_single, test_size=0.5)

        X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder,
                                                                               binarize=binarize,
                                                                               remove_structural=remove_structural)

        augmented_data = augment_splitted_data(X_train, y_train, X_calib, y_calib, None, None,
                                                            y_nhot_mixtures, n_celltypes, n_features,
                                                            label_encoder, priors0, priors1, [binarize],
                                                            [n_samples_per_combination]*3,
                                                            disallowed_mixtures=disallowed_mixtures)

        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(augmented_data.y_train_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T
        specific_model = clf_with_correct_settings('MLR', softmax=False, n_classes=-1, with_calibration=True)
        specific_model.fit_classifier(augmented_data.X_train_augmented, y_train)
        specific_model.fit_calibration(augmented_data.X_calib_augmented, augmented_data.y_calib_nhot_augmented, target_classes)

        log_lrs = []
        specific_log_lrs = []
        for sample in samples_to_evaluate:
            log_lrs.append(np.log10(model.predict_lrs([sample], target_classes))[0][-1])
            specific_log_lrs.append(np.log10(specific_model.predict_lrs([sample], target_classes))[0][-1])
        plot_multiclass_comparison(specific_log_lrs,
                                   log_lrs,
                                   ['blood+nas+vag', 'menstr','indication of menstr', 'blood', 'semen'],
                                   'specific_hypothesis',
                                   save_path, x_label='log(LR)', y_label='log(LR) H2: blood')

    compare_to_multiclass(X_single, y_single, target_classes, tc, model,
                          samples_to_evaluate, save_path=save_path, alternative_target=None)

    # plot the coefficients
    plot_coefficient_importances(
        model, target_classes, present_markers, label_encoder,
        savefig=os.path.join(save_path, 'coefs_{}'.format(model_name)), show=None)

    for t in range(len(target_classes)):
        intercept, coefficients = model.get_coefficients(t, target_classes[t].squeeze())
        all_coefficients = np.append(intercept, coefficients).tolist()
        all_coefficients_str = [str(coef) for coef in all_coefficients]
        all_coefficients_strr = [coef.replace('.', ',') for coef in all_coefficients_str]
        present_markers.insert(0, 'intercept')

        with open(os.path.join(save_path, 'coefs_{}_{}.csv'.format(tc[t].replace('/', '_'), model_name)), mode='w') as coefs:
            coefs_writer = csv.writer(coefs, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            coefs_writer.writerow(present_markers)
            coefs_writer.writerow(all_coefficients_strr)


def nfold_analysis(nfolds, tc, savepath, models_list, softmax_list: List[bool],
                   priors0: List[str], priors1: List[str], binarize_list: List[bool], test_size: float, calibration_size: float,
                   remove_structural: bool, calibration_on_loglrs: bool, nsamples: Tuple[int, int, int]):

    mle = MultiLabelEncoder(len(single_cell_types))

    baseline_prior = (tuple(priors0), tuple(priors1))
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
        emtpy_numpy_array = np.zeros((len(binarize_list), len(softmax_list), len(models_list), 1))
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
            coeffs[target_class_str] = np.zeros((len(binarize_list),1,X_single[0].shape[1]+1, 1))
        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=calibration_size)

        for i, binarize in enumerate(binarize_list):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, remove_structural=remove_structural)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            augmented_data[baseline_prior] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                y_nhot_mixtures, n_celltypes, n_features,
                                                                label_encoder,
                                                                priors0=priors0,
                                                                priors1=priors1,
                                                                binarize=binarize,
                                                                nsamples=nsamples, disallowed_mixtures=None)


            # ======= Transform test data as well =======
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
                        p=0
                        str_prior = baseline_prior
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
                            intercept, coefficients = model[str_prior].get_coefficients(t, target_class)
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


def compare_to_multiclass(X_single, y_single, target_classes, tc,
                          model: MarginalClassifier, samples,
                          binarize=True, save_path=None, alternative_target=None):
    """
    trains a multiclass model and compares its output to the given
    multilabel model, on list of samples
    """
    X = binarize_and_combine_samples(X_single, binarize)

    multi_class_model = LogisticRegression()
    multi_class_model.fit(X, y_single)
    for sample in samples:
        sample= np.array(sample).reshape(1, -1)
        multi_pred_single = multi_class_model.predict_proba(sample)[0]
        multi_pred_tc = []
        for target_class in target_classes:
            prob_any = np.sum(multi_pred_single[target_class == 1])
            prob_max_not = np.max(multi_pred_single[target_class == 0])
            if alternative_target is not None:
                prob_max_not = np.sum(multi_pred_single[alternative_target[0] == 1])
            multi_pred_tc.append(prob_any/prob_max_not)
        multi_pred_tc=np.array(multi_pred_tc)
        multi_log_lrs = np.log10(multi_pred_tc)
        # alternatively, we do not do the max trick but divide by 1-p. This
        # give indistinguishable results:
        # multi_log_lrs = np.log10(multi_pred_tc/(1-multi_pred_tc))

        # take single cell target classes
        log_lrs = np.log10(model.predict_lrs(sample, target_classes))
        plot_multiclass_comparison(log_lrs[0], multi_log_lrs, tc, sample, save_path)


def makeplots(tc, path, savepath, remove_structural: bool, nfolds, binarize_list, softmax_list, models_list, **kwargs):

    _, _, _, _, _, label_encoder, _, _ = \
        get_data_per_cell_type(single_cell_types=single_cell_types, remove_structural=remove_structural)
    target_classes = string2vec(tc, label_encoder)

    lrs_for_model_per_fold = OrderedDict()
    emtpy_numpy_array = np.zeros(
        (nfolds, len(binarize_list), len(softmax_list), len(models_list), 1))
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
        coeffs[target_class_str] = np.zeros((nfolds, len(binarize_list),1, len(marker_names)-4+1, 1))

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

    types_data = ['test augm', 'mixt']

    for type_data in types_data:
        lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = append_lrs_for_all_folds(
            lrs_for_model_per_fold, type=type_data)

        # plot_pavs_all_methods(lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods,
        #                           target_classes, label_encoder, savefig=os.path.join(savepath, 'pav_{}'.format(type_data)))

        for kind in ['roc', 'histogram']:
            plot_property_all_lrs_all_folds(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes,
                                        label_encoder, kind=kind,
                                        savefig=os.path.join(savepath, f'{kind}_{type_data}'))


    lrs_before_for_all_methods, lrs_after_for_all_methods, \
    y_nhot_for_all_methods = append_lrs_for_all_folds(
        lrs_for_model_per_fold, type='test augm')
    if nfolds > 1:
        for t, target_class in enumerate(target_classes):
            target_class_str = vec2string(target_class, label_encoder)
            target_class_save = target_class_str.replace(" ", "_")
            target_class_save = target_class_save.replace(".", "_")
            target_class_save = target_class_save.replace("/", "_")


            plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_test[target_class_str], label_encoder, "$C_{llr}$",
                                   savefig=os.path.join(savepath, 'boxplot_cllr_test_{}'.format(target_class_save)))
            plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                                   savefig=os.path.join(savepath, 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
            if DEBUG:
                plot_boxplot_of_metric(binarize_list, softmax_list, models_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                                       savefig=os.path.join(savepath, 'boxplot_accuracy_train_{}'.format(target_class_save)))
                plot_boxplot_of_metric(binarize_list, softmax_list, models_list, accuracies_test[target_class_str], label_encoder, "accuracy",
                                       savefig=os.path.join(savepath, 'boxplot_accuracy_test_{}'.format(target_class_save)))
                plot_boxplot_of_metric(binarize_list, softmax_list, models_list, cllr_test_as_mixtures[target_class_str], label_encoder, "$C_{llr}$",
                                       savefig=os.path.join(savepath, 'boxplot_cllr_test_as_mixt_{}'.format(target_class_save)))
                plot_progress_of_metric(binarize_list, softmax_list, models_list, accuracies_train[target_class_str], label_encoder, 'accuracy',
                                        savefig=os.path.join(savepath, 'progress_accuracy_train_{}'.format(target_class_save)))
                plot_progress_of_metric(binarize_list, softmax_list, models_list, accuracies_test[target_class_str], label_encoder, 'accuracy',
                                        savefig=os.path.join(savepath, 'progress_accuracy_test_{}'.format(target_class_save)))
                plot_progress_of_metric(binarize_list, softmax_list, models_list, cllr_test[target_class_str], label_encoder, '$C_{llr}$',
                                        savefig=os.path.join(savepath, 'progress_cllr_test_{}'.format(target_class_save)))
                plot_boxplot_of_metric(binarize_list, [False], [[a, True] for a in ['intercept']+marker_names], coeffs[target_class_str], label_encoder, "log LR",
                                   savefig=os.path.join(savepath, 'boxplot_coefficients_{}'.format(target_class_save)), ylim=[-3,3])


"""

"""

import os
import pickle
import csv

import numpy as np
import matplotlib.pyplot as plt
import rna.settings as settings

from collections import OrderedDict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, append_lrs_for_all_folds
from rna.augment import MultiLabelEncoder, augment_splitted_data, augment_data
from rna.constants import single_cell_types, marker_names
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax
from rna.plotting import plot_scatterplots_all_lrs_different_priors, plot_boxplot_of_metric, \
    plot_histograms_all_lrs_all_folds, plot_progress_of_metric, plot_rocs, plot_pavs_all_methods, \
    plot_coefficient_importance
from rna.lr_system import MarginalMLRClassifier


def get_trained_mlr_model(tc, retrain, n_samples_per_combination, binarize, from_penile, model_name):
    mle = MultiLabelEncoder(len(single_cell_types))

    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=False)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)

    if retrain:
        model = MarginalMLRClassifier()
        X_augmented, y_nhot_augmented = augment_data(X_single, y_single, n_celltypes, n_features,
                                                     n_samples_per_combination, label_encoder, prior=None,
                                                     binarize=binarize, from_penile=from_penile)

        indices = [np.argwhere(target_classes[i, :] == 1).flatten().tolist() for i in range(target_classes.shape[0])]
        y_train = np.array([np.max(np.array(y_nhot_augmented[:, indices[i]]), axis=1) for i in range(len(indices))]).T

        model.fit_classifier(X_augmented, y_train)
        pickle.dump(model, open('{}'.format(model_name), 'wb'))
    else:
        model = pickle.load(open('{}'.format(model_name), 'rb'))

    intercept = model._classifier.intercept_
    coefficients = model._classifier.coef_

    # plot the coefficients
    for list_object in tc:
        celltype = list_object.split(' and/or ')
    plot_coefficient_importance(intercept, coefficients, present_markers, celltype)
    plt.tight_layout()
    plt.savefig('coefs_{}'.format(model_name))
    plt.close()

    all_coefficients = np.append(intercept, coefficients).tolist()
    all_coefficients_str = [str(coef) for coef in all_coefficients]
    all_coefficients_strr = [coef.replace('.', ',') for coef in all_coefficients_str]
    present_markers.insert(0, 'intercept')

    with open('coefs_{}.csv'.format(model_name), mode='w') as coefs:
        coefs_writer = csv.writer(coefs, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        coefs_writer.writerow(present_markers)
        coefs_writer.writerow(all_coefficients_strr)


def nfold_analysis(nfolds, tc, savepath):
    if settings.from_penile == True:
        if True in settings.softmax:
            raise ValueError("The results following from these settings have not been validated and hence cannot be "
                             "relied on. Make sure 'softmax' is set to False if 'from_penile' is {}".format(
                settings.from_penile))
        for models_and_calib in settings.models:
            if 'MLP' in models_and_calib or 'XGB' in models_and_calib or 'DL' in models_and_calib:
                raise ValueError("The results following from these settings have not validated and hence cannot be "
                                 "relied on. The model cannot be {} if 'from_penile' is {}. Either adjust the model "
                                 "to 'MLR' or set 'from_penile=False'.".format(models_and_calib[0], settings.from_penile))

    mle = MultiLabelEncoder(len(single_cell_types))
    baseline_prior = str(settings.priors[0])

    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)


    if settings.split_before:
        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

    outer = tqdm(total=nfolds, desc='{} folds'.format(nfolds), position=0, leave=False)
    for n in range(nfolds):
        # n = n + (nfolds * run)
        print(n)

        # ======= Initialize =======
        lrs_for_model_in_fold = OrderedDict()
        emtpy_numpy_array = np.zeros((len(settings.binarize), len(settings.softmax), len(settings.models), len(settings.priors)))
        accuracies_train_n, accuracies_test_n, accuracies_test_as_mixtures_n, accuracies_mixtures_n, accuracies_single_n,\
        cllr_test_n, cllr_test_as_mixtures_n, cllr_mixtures_n = [dict() for i in range(8)]

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

        if not settings.split_before:
            # ======= Split data =======
            X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
            X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        for i, binarize in enumerate(settings.binarize):
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(settings.priors):
                augmented_data[str(priors)] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                    y_nhot_mixtures, n_celltypes, n_features,
                                                                    label_encoder, AugmentedData, priors, binarize,
                                                                    settings.from_penile)

            # ======= Transform data accordingly =======
            if binarize:
                X_test_transformed = [
                    [np.where(X_test[i][j] > 150, 1, 0) for j in range(len(X_test[i]))] for i in
                    range(len(X_test))]
                X_test_transformed = combine_samples(np.array(X_test_transformed))
            else:
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                for k, models in enumerate(settings.models):
                    print(models[0])


                    # ======= Calculate LRs before and after calibration =======
                    key_name = bool2str_binarize(binarize) + '_' + bool2str_softmax(softmax) + '_' + str(models[0])
                    if models[1]:
                        key_name+='_cal'
                    key_name_per_fold = str(n) + '_' + key_name
                    model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                    lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
                    lrs_before_calib_mixt, lrs_after_calib_mixt = \
                        calculate_lrs_for_different_priors(augmented_data, X_mixtures, target_classes, baseline_prior,
                                                           present_markers, models, mle, label_encoder, key_name_per_fold,
                                                           softmax, settings.calibration_on_loglrs, savepath)

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
                        for p, priors in enumerate(settings.priors):
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


def makeplots(nfolds, tc, path, savepath):

    _, _, _, _, _, label_encoder, _, _ = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    target_classes = string2vec(tc, label_encoder)

    lrs_for_model_per_fold = OrderedDict()
    emtpy_numpy_array = np.zeros(
        (nfolds, len(settings.binarize), len(settings.softmax), len(settings.models), len(settings.priors)))
    accuracies_train, accuracies_test, accuracies_test_as_mixtures, accuracies_mixtures, accuracies_single, \
    cllr_test, cllr_test_as_mixtures, cllr_mixtures = [dict() for i in range(8)]

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

    types_data = ['test augm', 'mixt']  # 'mixt' and/or 'test augm as mixt'

    for type_data in types_data:
        lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = append_lrs_for_all_folds(
            lrs_for_model_per_fold, type=type_data)

        plot_pavs_all_methods(lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods,
                                  target_classes, label_encoder, savefig=os.path.join(savepath, 'pav_{}'.format(type_data)))

        # plot_rocs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder)
                  # savefig=os.path.join(savepath, 'roc_{}'.format(type_data)))

        plot_histograms_all_lrs_all_folds(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes,
                                          label_encoder,
                                          savefig=os.path.join(savepath, 'histograms_after_calib_{}'.format(type_data)))

        if len(settings.priors) == 2:
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

        plot_boxplot_of_metric(accuracies_train[target_class_str], label_encoder, 'accuracy',
                               savefig=os.path.join(savepath, 'boxplot_accuracy_train_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_train[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_train_{}'.format(target_class_save)))

        plot_boxplot_of_metric(accuracies_test[target_class_str], label_encoder, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_test_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_test[target_class_str], label_encoder, 'accuracy',
                                savefig=os.path.join(savepath, 'progress_accuracy_test_{}'.format(target_class_save)))
    #
    #     plot_boxplot_of_metric(accuracies_test_as_mixtures[target_class_str], label_encoder, "accuracy",
    #                            savefig=os.path.join(savepath,
    #                                                 'boxplot_accuracy_test_as_mixtures_{}'.format(target_class_save)))
    #     plot_progress_of_metric(accuracies_test_as_mixtures[target_class_str], label_encoder, 'accuracy',
    #                             savefig=os.path.join(savepath,
    #                                                  'progress_accuracy_test_as_mixtures_{}'.format(target_class_save)))
    #
    #     plot_boxplot_of_metric(accuracies_mixtures[target_class_str], label_encoder, "accuracy",
    #                            savefig=os.path.join(savepath, 'boxplot_accuracy_mixtures_{}'.format(target_class_save)))
    #     plot_progress_of_metric(accuracies_mixtures[target_class_str], label_encoder, 'accuracy',
    #                             savefig=os.path.join(savepath,
    #                                                  'progress_accuracy_mixtures_{}'.format(target_class_save)))
    #
    #     plot_boxplot_of_metric(accuracies_single[target_class_str], label_encoder, "accuracy",
    #                            savefig=os.path.join(savepath, 'boxplot_accuracy_single_{}'.format(target_class_save)))
    #     plot_progress_of_metric(accuracies_single[target_class_str], label_encoder, 'accuracy',
    #                             savefig=os.path.join(savepath, 'progress_accuracy_single_{}'.format(target_class_save)))
    #
        plot_boxplot_of_metric(cllr_test[target_class_str], label_encoder, "Cllr",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_progress_of_metric(cllr_test[target_class_str], label_encoder, 'Cllr',
                                savefig=os.path.join(savepath, 'progress_cllr_test_{}'.format(target_class_save)))
    #
    #     plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], label_encoder, "Cllr",
    #                            savefig=os.path.join(savepath,
    #                                                 'boxplot_cllr_test_as_mixtures_{}'.format(target_class_save)))
    #     plot_progress_of_metric(cllr_test_as_mixtures[target_class_str], label_encoder, 'Cllr',
    #                             savefig=os.path.join(savepath,
    #                                                  'progress_cllr_test_as_mixtures_{}'.format(target_class_save)))
    #
    #     plot_boxplot_of_metric(cllr_mixtures[target_class_str], label_encoder, "Cllr",
    #                            savefig=os.path.join(savepath, 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
    #     plot_progress_of_metric(cllr_mixtures[target_class_str], label_encoder, 'Cllr',
    #                             savefig=os.path.join(savepath, 'progress_cllr_mixtures_{}'.format(target_class_save)))


# TODO: Want to change to dict?
class AugmentedData():

    def __init__(self, X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
           X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented, y_test_as_mixtures_nhot_augmented):
        self.X_train_augmented = X_train_augmented
        self.y_train_nhot_augmented = y_train_nhot_augmented
        self.X_calib_augmented = X_calib_augmented
        self.y_calib_nhot_augmented = y_calib_nhot_augmented
        self.X_test_augmented = X_test_augmented
        self.y_test_nhot_augmented = y_test_nhot_augmented
        self.X_test_as_mixtures_augmented = X_test_as_mixtures_augmented
        self.y_test_as_mixtures_nhot_augmented = y_test_as_mixtures_nhot_augmented

# TODO: Want to change to dict?
class LrsBeforeAfterCalib():

    def __init__(self, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, lrs_before_calib_test_as_mixtures,
                 lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, lrs_before_calib_mixt,
                 lrs_after_calib_mixt, y_mixtures_nhot):
        self.lrs_before_calib = lrs_before_calib
        self.lrs_after_calib = lrs_after_calib
        self.y_test_nhot_augmented = y_test_nhot_augmented
        self.lrs_before_calib_test_as_mixtures = lrs_before_calib_test_as_mixtures
        self.lrs_after_calib_test_as_mixtures = lrs_after_calib_test_as_mixtures
        self.y_test_as_mixtures_nhot_augmented = y_test_as_mixtures_nhot_augmented
        self.lrs_before_calib_mixt = lrs_before_calib_mixt
        self.lrs_after_calib_mixt = lrs_after_calib_mixt
        self.y_mixtures_nhot = y_mixtures_nhot
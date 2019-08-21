"""

"""

import os

import numpy as np
import rna.settings as settings

from collections import OrderedDict

from sklearn.model_selection import train_test_split

from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, append_lrs_for_all_folds
from rna.augment import MultiLabelEncoder, augment_splitted_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax
from rna.plotting import plot_scatterplots_all_lrs_different_priors, plot_boxplot_of_metric, \
    plot_histograms_all_lrs_all_folds, plot_progress_of_metric, plot_rocs, plot_per_feature


def nfold_analysis(nfolds, tc, savepath):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))
    baseline_prior = str(settings.priors[0])


    # ======= Load data =======
    X_single, y_nhot_single, n_celltypes, n_features, n_per_celltype, label_encoder, present_markers, present_celltypes = \
        get_data_per_cell_type(single_cell_types=single_cell_types, markers=settings.markers)
    y_single = mle.transform_single(mle.nhot_to_labels(y_nhot_single))
    target_classes = string2vec(tc, label_encoder)


    # ======= Initialize =======
    emtpy_numpy_array = np.zeros((nfolds, len(settings.binarize), len(settings.softmax), len(settings.models), len(settings.priors)))
    accuracies_train, accuracies_test, accuracies_test_as_mixtures, accuracies_mixtures, accuracies_single,\
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

    if settings.split_before:
        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)


    lrs_for_model_per_fold = OrderedDict()
    for n in range(nfolds):
        print("Fold {}".format(n))

        if not settings.split_before:
            # ======= Split data =======
            X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
            X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        lrs_for_model_in_fold = OrderedDict()
        for i, binarize in enumerate(settings.binarize):
            print(" Binarize the data: {} {}".format(binarize, i))
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(settings.priors):
                print(" Priors for augmenting data: {}".format(priors))

                augmented_data[str(priors)] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                    y_nhot_mixtures, n_celltypes, n_features,
                                                                    label_encoder, AugmentedData, priors, binarize,
                                                                    from_penile)

            # ======= Transform data accordingly =======
            if binarize:
                X_test_transformed = [
                    [np.where(X_test[i][j] > 150, 1, 0) for j in range(len(X_test[i]))] for i in
                    range(len(X_test))]
                X_test_transformed = combine_samples(np.array(X_test_transformed))
            else:
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                print(" Use softmax to calculate probabilities with: {} {}".format(softmax, j))
                for k, models in enumerate(settings.models):
                    print(" Model: {} {}".format(models[0], k))


                    # ======= Calculate LRs before and after calibration =======
                    key_name_per_fold = str(n) + '_' + bool2str_binarize(binarize) + '_' + bool2str_softmax(softmax) + '_' + str(models[0])
                    if settings.augment:
                        model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                        lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
                        lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(augmented_data,
                                                                                                         X_mixtures,
                                                                                                         target_classes,
                                                                                                         baseline_prior,
                                                                                                         present_markers,
                                                                                                         models, mle,
                                                                                                         label_encoder,
                                                                                                         key_name_per_fold,
                                                                                                         softmax,
                                                                                                         settings.calibration_on_loglrs)
                    else:
                        raise ValueError("There is no option to set settings.augment = {}".format(settings.augment))

                    key_name = bool2str_binarize(binarize) + '_' + bool2str_softmax(softmax) + '_' + str(models[0])
                    lrs_for_model_in_fold[key_name] = LrsBeforeAfterCalib(lrs_before_calib, lrs_after_calib, y_test_nhot_augmented,
                                                                          lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented,
                                                                          lrs_before_calib_mixt, lrs_after_calib_mixt, y_nhot_mixtures)

                    # ======= Make plot for OvR Logistic Regression =======
                    # if models[0] == 'MLR' and softmax == False:
                    #     plot_per_feature(model, augmented_data, target_classes, present_markers)
                    #     plot_per_feature(model, augmented_data, target_classes, present_markers, train=False)



                    # ======= Calculate performance metrics =======
                    for t, target_class in enumerate(target_classes):
                        for p, priors in enumerate(settings.priors):
                            str_prior = str(priors)
                            target_class_str = vec2string(target_class, label_encoder)

                            accuracies_train[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[str_prior].X_train_augmented,
                                augmented_data[str_prior].y_train_nhot_augmented, target_classes, model[str_prior],
                                mle)[t]
                            accuracies_test[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_augmented,
                                augmented_data[baseline_prior].y_test_nhot_augmented, target_classes, model[str_prior],
                                mle)[t]
                            accuracies_test_as_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                augmented_data[baseline_prior].X_test_as_mixtures_augmented,
                                augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_classes,
                                model[str_prior], mle)[t]
                            accuracies_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_mixtures, y_nhot_mixtures, target_classes, model[str_prior], mle)[t]
                            accuracies_single[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                X_test_transformed, mle.inv_transform_single(y_test), target_classes, model[str_prior],
                                mle)[t]

                            cllr_test[target_class_str][n, i, j, k, p] = cllr(
                                lrs_after_calib[str_prior][:, t], augmented_data[baseline_prior].y_test_nhot_augmented, target_class)
                            cllr_test_as_mixtures[target_class_str][n, i, j, k, p] = cllr(
                                lrs_after_calib_test_as_mixtures[str_prior][:, t], augmented_data[baseline_prior].y_test_as_mixtures_nhot_augmented, target_class)
                            cllr_mixtures[target_class_str][n, i, j, k, p] = cllr(
                                lrs_after_calib_mixt[str_prior][:, t], y_nhot_mixtures, target_class)
        lrs_for_model_per_fold[str(n)] = lrs_for_model_in_fold

    lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = append_lrs_for_all_folds(lrs_for_model_per_fold, type='test augm')
    plot_rocs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
              savefig=os.path.join(savepath, 'roc_curves'))

    plot_histograms_all_lrs_all_folds(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                      savefig=os.path.join(savepath, 'histograms_after_calib_augm'))
    if len(settings.priors) == 2:
        plot_scatterplots_all_lrs_different_priors(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                                   savefig=os.path.join(savepath, 'LRs_for_different_priors_augm'))

    # lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold, type='test augm as mixt')
    # plot_histogram_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
    #                        savefig=os.path.join('scratch', 'histograms_after_calib_augmasmixt'))
    # if len(settings.priors) == 2:
    #     plot_scatterplot_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
    #                              savefig=os.path.join('scratch', 'LRs_for_different_priors_augmasmixt'))
    #
    # lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold, type='mixt')
    # plot_histogram_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
    #                        savefig=os.path.join('scratch', 'histograms_after_calib_mixt'))
    # if len(settings.priors) == 2:
    #     plot_scatterplot_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
    #                              savefig=os.path.join('scratch', 'LRs_for_different_priors_mixt'))

    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plot_boxplot_of_metric(accuracies_train[target_class_str], label_encoder, target_class, 'accuracy',
                               savefig=os.path.join(savepath, 'boxplot_accuracy_train_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_train[target_class_str], label_encoder, target_class, 'accuracy',
                                savefig = os.path.join(savepath, 'progress_accuracy_train_{}'.format(target_class_save)))

        plot_boxplot_of_metric(accuracies_test[target_class_str], label_encoder, target_class, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_test_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_test[target_class_str], label_encoder, target_class, 'accuracy',
                                savefig = os.path.join(savepath, 'progress_accuracy_test_{}'.format(target_class_save)))

        plot_boxplot_of_metric(accuracies_test_as_mixtures[target_class_str], label_encoder, target_class, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_test_as_mixtures_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_test_as_mixtures[target_class_str], label_encoder, target_class, 'accuracy',
                                savefig = os.path.join(savepath, 'progress_accuracy_test_as_mixtures_{}'.format(target_class_save)))

        plot_boxplot_of_metric(accuracies_mixtures[target_class_str], label_encoder, target_class, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_mixtures_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_mixtures[target_class_str], label_encoder, target_class, 'accuracy',
                                savefig = os.path.join(savepath, 'progress_accuracy_mixtures_{}'.format(target_class_save)))

        plot_boxplot_of_metric(accuracies_single[target_class_str], label_encoder, target_class, "accuracy",
                               savefig=os.path.join(savepath, 'boxplot_accuracy_single_{}'.format(target_class_save)))
        plot_progress_of_metric(accuracies_single[target_class_str], label_encoder, target_class, 'accuracy',
                                savefig = os.path.join(savepath, 'progress_accuracy_single_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_test[target_class_str], label_encoder, target_class, "Cllr",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_progress_of_metric(cllr_test[target_class_str], label_encoder, target_class, 'Cllr',
                                savefig = os.path.join(savepath, 'progress_cllr_test_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], label_encoder, target_class, "Cllr",
                               savefig=os.path.join(savepath, 'boxplot_cllr_test_as_mixtures_{}'.format(target_class_save)))
        plot_progress_of_metric(cllr_test_as_mixtures[target_class_str], label_encoder, target_class, 'Cllr',
                                savefig = os.path.join(savepath, 'progress_cllr_test_as_mixtures_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_mixtures[target_class_str], label_encoder, target_class, "Cllr",
                               savefig=os.path.join(savepath, 'boxplot_cllr_mixtures_{}'.format(target_class_save)))
        plot_progress_of_metric(cllr_mixtures[target_class_str], label_encoder, target_class, 'Cllr',
                                savefig = os.path.join(savepath, 'progress_cllr_mixtures_{}'.format(target_class_save)))

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
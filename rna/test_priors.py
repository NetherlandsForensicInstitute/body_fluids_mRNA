"""

"""

import os

import numpy as np
import rna.settings as settings

from collections import OrderedDict

from sklearn.model_selection import train_test_split

from rna.analytics import combine_samples, calculate_accuracy_all_target_classes, cllr, \
    calculate_lrs_for_different_priors, combine_lrs_for_all_folds
from rna.augment import MultiLabelEncoder, augment_splitted_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.utils import vec2string, string2vec, bool2str_binarize, bool2str_softmax
from rna.plotting import plot_scatterplot_all_lrs_before_after_calib, plot_scatterplot_all_lrs, \
    plot_boxplot_of_metric, plot_histogram_all_lrs


def test_priors(nfolds, tc):
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

    lrs_for_model_per_fold = OrderedDict()
    for n in range(nfolds):
        print("Fold {}".format(n))

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
                print("Priors for augmenting data: {}".format(priors))

                augmented_data[str(priors)] = augment_splitted_data(X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                                    y_nhot_mixtures, n_celltypes, n_features,
                                                                    label_encoder, AugmentedData, priors, binarize,
                                                                    from_penile)

            # ======= Transform data accordingly =======
            if binarize:
                # X_train_transformed = np.where(combine_samples(X_train) > 150, 1, 0)
                # X_calib_transformed = np.where(combine_samples(X_calib) > 150, 1, 0)
                X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            else:
                # X_train_transformed = combine_samples(X_train) / 1000
                # X_calib_transformed = combine_samples(X_calib) / 1000
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                print(" Use softmax to calculate probabilities with: {} {}".format(softmax, j))
                for k, models in enumerate(settings.models):
                    print(" Model: {} {}".format(models[0], k))


                    # ======= Calculate LRs before and after calibration =======
                    if settings.augment:
                        model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                        lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented, \
                        lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(augmented_data,
                                                                                                         X_mixtures,
                                                                                                         target_classes,
                                                                                                         baseline_prior,
                                                                                                         models, mle,
                                                                                                         label_encoder,
                                                                                                         softmax,
                                                                                                         settings.calibration_on_loglrs)

                    key_name = bool2str_binarize(binarize) + '_' + bool2str_softmax(softmax) + '_' + str(models[0])
                    lrs_for_model_in_fold[key_name] = LrsBeforeAfterCalib(lrs_before_calib, lrs_after_calib, y_test_nhot_augmented,
                                                                          lrs_before_calib_test_as_mixtures, lrs_after_calib_test_as_mixtures, y_test_as_mixtures_nhot_augmented,
                                                                          lrs_before_calib_mixt, lrs_after_calib_mixt, y_nhot_mixtures)

                    # TODO: Check whether want to include --> original data
                    # else:
                    #     y_train_transformed = mle.inv_transform_single(y_train)
                    #     y_train_transformed = mle.labels_to_nhot(y_train_transformed)
                    #     y_calib_transformed = mle.inv_transform_single(y_calib)
                    #     y_calib_transformed = mle.labels_to_nhot(y_calib_transformed)
                    #     model, lrs_before_calib, lrs_after_calib, lrs_before_calib_test_as_mixtures, \
                    #     lrs_after_calib_test_as_mixtures, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                    #         perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_transformed,
                    #                          y_train_transformed, X_calib_transformed, y_calib_transformed, X_test_augmented,
                    #                          y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes, save_hist=True)


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

    lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold, type='test augm')
    plot_histogram_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                           savefig=os.path.join('scratch', 'histograms_after_calib_augm'))
    # plot_scatterplot_all_lrs_before_after_calib(lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder)
    if len(settings.priors) == 2:
        plot_scatterplot_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                 savefig=os.path.join('scratch', 'LRs_for_different_priors_augm'))

    lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold, type='test augm as mixt')
    plot_histogram_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                           savefig=os.path.join('scratch', 'histograms_after_calib_augmasmixt'))
    if len(settings.priors) == 2:
        plot_scatterplot_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                 savefig=os.path.join('scratch', 'LRs_for_different_priors_augmasmixt'))

    lrs_before_for_all_methods, lrs_after_for_all_methods, y_nhot_for_all_methods = combine_lrs_for_all_folds(lrs_for_model_per_fold, type='mixt')
    plot_histogram_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                           savefig=os.path.join('scratch', 'histograms_after_calib_mixt'))
    if len(settings.priors) == 2:
        plot_scatterplot_all_lrs(lrs_after_for_all_methods, y_nhot_for_all_methods, target_classes, label_encoder,
                                 savefig=os.path.join('scratch', 'LRs_for_different_priors_mixt'))

    for t, target_class in enumerate(target_classes):
        target_class_str = vec2string(target_class, label_encoder)
        target_class_save = target_class_str.replace(" ", "_")
        target_class_save = target_class_save.replace(".", "_")
        target_class_save = target_class_save.replace("/", "_")

        plot_boxplot_of_metric(accuracies_train[target_class_str], "accuracy",
                               savefig=os.path.join('scratch', 'boxplot_accuracy_train_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_test[target_class_str], "accuracy",
                               savefig=os.path.join('scratch', 'boxplot_accuracy_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_test_as_mixtures[target_class_str], "accuracy",
                               savefig=os.path.join('scratch', 'boxplot_accuracy_test_as_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_mixtures[target_class_str], "accuracy",
                               savefig=os.path.join('scratch', 'boxplot_accuracy_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(accuracies_single[target_class_str], "accuracy",
                               savefig=os.path.join('scratch', 'boxplot_accuracy_single_{}'.format(target_class_save)))

        plot_boxplot_of_metric(cllr_test[target_class_str], "Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_{}'.format(target_class_save)))
        plot_boxplot_of_metric(cllr_test_as_mixtures[target_class_str], "Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_test_as_mixtures_{}'.format(target_class_save)))
        plot_boxplot_of_metric(cllr_mixtures[target_class_str], "Cllr",
                               savefig=os.path.join('scratch', 'boxplot_cllr_mixtures_{}'.format(target_class_save)))

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
"""

"""

import os

import numpy as np
import rna.settings as settings

from collections import OrderedDict

from sklearn.model_selection import train_test_split

from rna.analytics import cllr, combine_samples, calculate_accuracy_all_target_classes
from rna.augment import MultiLabelEncoder, augment_splitted_data
from rna.constants import single_cell_types
from rna.input_output import get_data_per_cell_type, read_mixture_data
from rna.lr_system import perform_analysis
from rna.utils import vec2string, string2vec
from rna.plotting import plot_boxplot_of_metric, plot_scatterplot_lrs


def test_priors(nfolds, tc):
    from_penile = False
    mle = MultiLabelEncoder(len(single_cell_types))


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


    for n in range(nfolds):
        print("Fold {}".format(n))

        # ======= Split data =======
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, stratify=y_single, test_size=settings.test_size)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, stratify=y_train, test_size=settings.calibration_size)

        lrs_for_model = OrderedDict()
        for i, binarize in enumerate(settings.binarize):
            print(" Binarize the data: {} {}".format(binarize, i))
            X_mixtures, y_nhot_mixtures, mixture_label_encoder = read_mixture_data(n_celltypes, label_encoder, binarize=binarize, markers=settings.markers)


            # ======= Augment data for all priors =======
            augmented_data = OrderedDict()
            for p, priors in enumerate(settings.priors):
                print("Priors for augmenting data: {}".format(priors))

                augmented_data[str(priors)] = augment_splitted_data(X_calib, X_test, X_train, binarize, from_penile,
                                                                    label_encoder, n_celltypes, n_features, priors,
                                                                    y_calib, y_nhot_mixtures, y_test, y_train, AugmentedData)

            # ======= Transform data accordingly =======
            if binarize:
                X_train_transformed = np.where(combine_samples(X_train) > 150, 1, 0)
                X_calib_transformed = np.where(combine_samples(X_calib) > 150, 1, 0)
                X_test_transformed = np.where(combine_samples(X_test) > 150, 1, 0)
            else:
                X_train_transformed = combine_samples(X_train) / 1000
                X_calib_transformed = combine_samples(X_calib) / 1000
                X_test_transformed = combine_samples(X_test) / 1000

            for j, softmax in enumerate(settings.softmax):
                print(" Use softmax to calculate probabilities with: {} {}".format(softmax, j))
                for k, models in enumerate(settings.models):
                    print(" Model: {} {}".format(models[0], k))

                    # ======= Calculate LRs before and after calibration =======
                    if settings.augment:
                        model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
                        lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib, y_test_as_mixtures_nhot_augmented, \
                        lrs_before_calib_mixt, lrs_after_calib_mixt = calculate_lrs_for_different_priors(
                            augmented_data, X_mixtures, n, binarize, softmax, models, mle, label_encoder, target_classes)

                    key_name = str(binarize) + '_' + str(softmax) + '_' + str(models[0])
                    lrs_for_model[key_name] = LrsAfterCalib(lrs_after_calib, y_test_nhot_augmented,
                                                            lrs_test_as_mixtures_after_calib, y_test_as_mixtures_nhot_augmented,
                                                            lrs_after_calib_mixt)

                    # TODO: Check whether want to include
                    # else:
                    #     y_train_transformed = mle.inv_transform_single(y_train)
                    #     y_train_transformed = mle.labels_to_nhot(y_train_transformed)
                    #     y_calib_transformed = mle.inv_transform_single(y_calib)
                    #     y_calib_transformed = mle.labels_to_nhot(y_calib_transformed)
                    #     model, lrs_before_calib, lrs_after_calib, lrs_test_as_mixtures_before_calib, \
                    #     lrs_test_as_mixtures_after_calib, lrs_before_calib_mixt, lrs_after_calib_mixt = \
                    #         perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_transformed,
                    #                          y_train_transformed, X_calib_transformed, y_calib_transformed, X_test_augmented,
                    #                          y_test_nhot_augmented, X_test_as_mixtures_augmented, X_mixtures, target_classes, save_hist=True)


                    # ======= Calculate performance metrics =======
                    for t, target_class in enumerate(target_classes):
                        for p, priors in enumerate(settings.priors):
                            str_prior = str(priors)
                            target_class_str = vec2string(target_class, label_encoder)

                            accuracies_train[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                model[str_prior], mle, augmented_data[str_prior].y_train_nhot_augmented,
                                augmented_data[str_prior].X_train_augmented, target_classes)[t]
                            accuracies_test[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                model[str_prior], mle, augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented,
                                augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_augmented, target_classes)[t]
                            accuracies_test_as_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                model[str_prior], mle, augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented,
                                augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_as_mixtures_augmented, target_classes)[t]
                            accuracies_mixtures[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                model[str_prior], mle, y_nhot_mixtures, X_mixtures, target_classes)[t]
                            accuracies_single[target_class_str][n, i, j, k, p] = calculate_accuracy_all_target_classes(
                                model[str_prior], mle, mle.inv_transform_single(y_test), X_test_transformed, target_classes)[t]

                            cllr_test[target_class_str][n, i, j, k, p] = cllr(
                                lrs_after_calib[str_prior][:, t], augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented, target_class)
                            cllr_test_as_mixtures[target_class_str][n, i, j, k, p] = cllr(
                                lrs_test_as_mixtures_after_calib[str_prior][:, t], augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented, target_class)
                            cllr_mixtures[target_class_str][n, i, j, k, p] = cllr(
                                lrs_after_calib_mixt[str_prior][:, t], y_nhot_mixtures, target_class)

        plot_scatterplot_lrs(lrs_for_model, label_encoder, y_nhot_mixtures, target_classes, savefig='LRs_for_different_priors')

    plot_boxplot_of_metric(accuracies_train['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_train'))
    plot_boxplot_of_metric(accuracies_test['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_test'))
    plot_boxplot_of_metric(accuracies_test_as_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_test_as_mixtures'))
    plot_boxplot_of_metric(accuracies_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_mixtures'))
    plot_boxplot_of_metric(accuracies_single['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "accuracy",
                           savefig=os.path.join('scratch', 'boxplot_accuracy_single'))

    plot_boxplot_of_metric(cllr_test['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_test'))
    plot_boxplot_of_metric(cllr_test_as_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_test_as_mixtures'))
    plot_boxplot_of_metric(cllr_mixtures['Menstrual.secretion and/or Vaginal.mucosa'][:, :, :, :, :], "Cllr",
                           savefig=os.path.join('scratch', 'boxplot_cllr_mixtures'))

# TODO: Want to change to dict?
class AugmentedData():

    def __init__(self, X_train_augmented, y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented, \
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
class LrsAfterCalib():

    def __init__(self, lrs_after_calib, y_test_nhot_augmented, lrs_test_as_mixtures_after_calib,
                 y_test_as_mixtures_nhot_augmented, lrs_after_calib_mixt):
        self.lrs_after_calib = lrs_after_calib
        self.y_test_nhot_augmented = y_test_nhot_augmented
        self.lrs_test_as_mixtures_after_calib = lrs_test_as_mixtures_after_calib
        self.y_test_as_mixtures_nhot_augmented = y_test_as_mixtures_nhot_augmented
        self.lrs_after_calib_mixt = lrs_after_calib_mixt


def prior2string(prior, label_encoder):

    # convert string into list of integers
    prior = prior.strip('][').split(', ')
    prior = [int(prior[i]) for i in range(len(prior))]

    if len(np.unique(prior)) == 1:
        return 'Uniform'

    else:
        counts = {prior.count(value): value for value in list(set(prior))}
        value_relevant_prior = counts[1]
        index_of_relevant_prior = prior.index(value_relevant_prior)
        counts.pop(1)
        value_other_priors = list(counts.values())[0]

        if value_relevant_prior > value_other_priors:
            difference = 'more'
            value = value_relevant_prior
        elif value_relevant_prior < value_other_priors:
            difference = 'less'
            value = value_other_priors

        name = label_encoder.inverse_transform([index_of_relevant_prior])[0]

        return '{} {}x {} likely'.format(name, value, difference)


def calculate_lrs_for_different_priors(augmented_data, X_mixtures, n, binarize, softmax, models, mle, label_encoder,
                                       target_classes):

    X_test_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_augmented
    y_test_nhot_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_nhot_augmented
    X_test_as_mixtures_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].X_test_as_mixtures_augmented
    y_test_as_mixtures_nhot_augmented = augmented_data['[1, 1, 1, 1, 1, 1, 1, 1]'].y_test_as_mixtures_nhot_augmented

    model = OrderedDict()
    lrs_before_calib = OrderedDict()
    lrs_after_calib = OrderedDict()
    lrs_test_as_mixtures_before_calib = OrderedDict()
    lrs_test_as_mixtures_after_calib = OrderedDict()
    lrs_before_calib_mixt = OrderedDict()
    lrs_after_calib_mixt = OrderedDict()

    for i, (key, data) in enumerate(augmented_data.items()):
        print(" Prior: {}".format(key))

        X_train_augmented = data.X_train_augmented
        y_train_nhot_augmented = data.y_train_nhot_augmented
        X_calib_augmented = data.X_calib_augmented
        y_calib_nhot_augmented = data.y_calib_nhot_augmented

        model_i, lrs_before_calib_i, lrs_after_calib_i, \
        lrs_test_as_mixtures_before_calib_i, lrs_test_as_mixtures_after_calib_i, \
        lrs_before_calib_mixt_i, lrs_after_calib_mixt_i = \
            perform_analysis(n, binarize, softmax, models, mle, label_encoder, X_train_augmented,
                             y_train_nhot_augmented, X_calib_augmented, y_calib_nhot_augmented,
                             X_test_augmented, y_test_nhot_augmented, X_test_as_mixtures_augmented,
                             X_mixtures, target_classes, name=key, save_hist=True)

        model[key] = model_i
        lrs_before_calib[key] = lrs_before_calib_i
        lrs_after_calib[key] = lrs_after_calib_i
        lrs_test_as_mixtures_before_calib[key] = lrs_test_as_mixtures_before_calib_i
        lrs_test_as_mixtures_after_calib[key] = lrs_test_as_mixtures_after_calib_i
        lrs_before_calib_mixt[key] = lrs_before_calib_mixt_i
        lrs_after_calib_mixt[key] = lrs_after_calib_mixt_i

    return model, lrs_before_calib, lrs_after_calib, y_test_nhot_augmented, \
           lrs_test_as_mixtures_before_calib, lrs_test_as_mixtures_after_calib,y_test_as_mixtures_nhot_augmented, \
           lrs_before_calib_mixt, lrs_after_calib_mixt
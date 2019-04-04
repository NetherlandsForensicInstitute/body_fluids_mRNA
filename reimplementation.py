from rna.lr_system import *
from rna.plotting import plot_for_experimental_mixture_data
from lir.plotting import makeplot_hist_density_avg, \
    makeplot_density_avg
from rna.utils import create_information_on_classes_to_evaluate, split_data, probs_to_lrs, average_per_celltype, \
    sort_calibrators, refactor_classes_map
from rna.analytics import *
from rna.input_output import *

if __name__ == '__main__':
    developing = False
    include_blank = False
    # TODO: Change classes_map, inv_classes_map names --> string2vec and vec2string
    X_raw_singles, y_raw_singles, n_single_cell_types, n_features, classes_map, inv_classes_map, n_per_class = \
        get_data_per_cell_type(developing=developing, include_blank=include_blank)
    # TODO: Make this function work
    #plot_data(X_raw_singles)
    n_folds = 2
    N_SAMPLES_PER_COMBINATION = 4
    MAX_LR = 10
    from_penile = False
    retrain = True
    model_file_name = 'mlpmodel'
    if from_penile:
        model_file_name+='_penile'

    # which classes should we compute marginals for? all single cell types and a 'contains vaginal' class?
    # '-1' to avoid the penile skin
    single_cell_classes = [inv_classes_map[j] for j in range(n_single_cell_types - 1)]
    classes_included = ['Menstrual.secretion', 'Nasal.mucosa', 'Saliva', 'Skin', 'Vaginal.mucosa']
    class_combinations_to_evaluate = [['Vaginal.mucosa', 'Menstrual.secretion']]
    class_combinations_to_evaluate_combined = [' and/or '.join(comb) for comb in class_combinations_to_evaluate]
    classes_to_evaluate = classes_included + class_combinations_to_evaluate_combined

    # extend original classes map with new combined classes
    classes_map_full, classes_map_to_evaluate = refactor_classes_map(
        classes_map,
        classes_to_evaluate,
        class_combinations_to_evaluate_combined,
        from_penile
    )

    # Split the data in two equal parts: for training and calibration
    X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
        split_data(X_raw_singles, y_raw_singles)

    if retrain:
        # NB penile skin treated like all others for classify_single
        classify_single(X_train, y_train, inv_classes_map)

        model = MarginalClassifier()
        # model = MLPClassifier(random_state=0)
        # model = LogisticRegression(random_state=0)

        h0_h1_lrs_all_log = []
        h0_h1_lrs_all = []
        h0_h1_lrs_all_after_log = []
        h0_h1_lrs_all_after = []

        h0_h1_probs_all_calibration = []
        all_calibrators_per_class = []

        for n in range(n_folds):
            print("Fold {}".format(n))

            # TODO this is not nfold, but independently random
            X_train, y_train, X_calibrate, y_calibrate, X_test, y_test = \
                split_data(X_raw_singles, y_raw_singles, size=(0.4, 0.4))

            # augment the train part of the data to train the MLP model on
            X_augmented_train, y_augmented_train, _, _ = \
                augment_data(
                    X_train,
                    y_train,
                    n_single_cell_types,
                    n_features,
                    N_SAMPLES_PER_COMBINATION,
                    classes_map,
                    from_penile=from_penile
            )

            print(
                'fitting on {} samples, {} features, {} classes'.format(
                    len(y_augmented_train),
                    X_augmented_train.shape[1],
                    len(set(y_augmented_train)))
            )

            # TODO get the mixture data from dorum

            model.fit(X_augmented_train, y_augmented_train)

            # augment calibration data to calibrate the model with
            X_calibrate_augmented, y_calibrate_augmented, y_augmented_matrix_calibrate, \
                mixture_classes_in_single_cell_type_calibration = \
                    augment_data(
                        X_calibrate,
                        y_calibrate,
                        n_single_cell_types,
                        n_features,
                        N_SAMPLES_PER_COMBINATION,
                        classes_map,
                        from_penile=from_penile
                    )

            # create information depending on the cell types of interest
            mixture_classes_in_classes_to_evaluate_calibration, classes_map_updated, \
                y_augmented_calibration_updated = create_information_on_classes_to_evaluate(
                mixture_classes_in_single_cell_type_calibration,
                classes_map_to_evaluate,
                class_combinations_to_evaluate,
                y_calibrate_augmented,
                y_augmented_matrix_calibrate
            )

            h0_h1_probs_calibration = model.predict_proba_per_class(
                X_calibrate_augmented,
                y_augmented_calibration_updated,
                mixture_classes_in_classes_to_evaluate_calibration,
                classes_map_updated,
                classes_map_full,
                MAX_LR
            )

            # augment test data to evaluate the model with
            X_augmented_test, y_augmented_test, y_augmented_matrix, mixture_classes_in_single_cell_type = \
                augment_data(
                    X_test,
                    y_test,
                    n_single_cell_types,
                    n_features,
                    4,
                    classes_map,
                    from_penile=from_penile
            )

            evaluate_model(model, 'test', X_augmented_test, y_augmented_test)

            # create information depending on the cell types of interest
            mixture_classes_in_classes_to_evaluate, _, y_augmented_test_updated = \
                create_information_on_classes_to_evaluate(
                    mixture_classes_in_single_cell_type,
                    classes_map_to_evaluate,
                    class_combinations_to_evaluate,
                    y_augmented_test,
                    y_augmented_matrix
            )

            h0_h1_probs_test = model.predict_proba_per_class(
                X_augmented_test,
                y_augmented_test_updated,
                mixture_classes_in_classes_to_evaluate,
                classes_map_updated,
                classes_map_full,
                MAX_LR
            )

            # fit calibrated models
            calibrators_per_class = calibration_fit(h0_h1_probs_calibration, classes_map_updated)

            # transform the test scores
            h0_h1_after_calibration = calibration_transform(h0_h1_probs_test, calibrators_per_class, classes_map_updated)

            # if n == 0:
            #     # only plot single class performance once
            #     # TODO: Make this function work
            #     # prob_per_class_test = model.predict_proba(
            #     #     X_calibrate_augmented,
            #     #     mixture_classes_in_classes_to_evaluate_calibration,
            #     #     classes_map_updated,
            #     #     MAX_LR
            #     # )
            #
            #     # Check correlation
            #     # print('Cor(Vag, VagMenstr):\n', np.corrcoef(prob_per_class_test[:, 4], prob_per_class_test[:, 5]))
            #     # print('Cor(Menstr, VagMenstr):\n', np.corrcoef(prob_per_class_test[:, 0], prob_per_class_test[:, 5]))
            #     # print('Cor(Vag+Menstr, VagMenstr):\n',
            #     #       np.corrcoef(np.sum([prob_per_class_test[:, 0], prob_per_class_test[:, 4]], axis=0), prob_per_class_test[:, 5]))
            #
            #     idxs = []
            #     for celltype in sorted(classes_map_updated):
            #         idxs.append(classes_map_full[celltype])
            #     y_augmented_test_relevant = y_augmented_test_updated[:, idxs]
            #
            #     # TODO: Make this plot work
            #     # boxplot_per_single_class_category(
            #     #     prob_per_class_test,
            #     #     y_augmented_test_relevant,
            #     #     classes_map_updated,
            #     #     class_combinations_to_evaluate
            #     # )

            h0_h1_lrs_all_log.append(probs_to_lrs(h0_h1_probs_test, classes_map_updated, log=True))
            h0_h1_lrs_all.append(probs_to_lrs(h0_h1_probs_test, classes_map_updated))
            h0_h1_lrs_all_after_log.append(probs_to_lrs(h0_h1_after_calibration, classes_map_updated, log=True))
            h0_h1_lrs_all_after.append(probs_to_lrs(h0_h1_after_calibration, classes_map_updated))
            h0_h1_probs_all_calibration.append(h0_h1_probs_calibration)
            all_calibrators_per_class.append(calibrators_per_class)

            # if n == 0:
            #     makeplot_hist_density(h0_h1_probs_calibration, calibrators_per_class, show=True)

        idxs = []
        for celltype in sorted(classes_map_updated):
            idxs.append(classes_map_full[celltype])
        y_augmented_test_relevant = y_augmented_test_updated[:, idxs]
        y = np.sort(y_augmented_test_relevant, axis=0)[::-1]

        h0_h1_lrs_avg_log = average_per_celltype(h0_h1_lrs_all_log)
        h0_h1_lrs_avg = average_per_celltype(h0_h1_lrs_all)
        h0_h1_lrs_avg_after_log = average_per_celltype(h0_h1_lrs_all_after_log)
        h0_h1_lrs_avg_after = average_per_celltype(h0_h1_lrs_all_after)
        h0_h1_probs_avg_calibration = average_per_celltype(h0_h1_probs_all_calibration)

        sorted_calibrators_per_class = sort_calibrators(all_calibrators_per_class)
        makeplot_hist_density_avg(h0_h1_probs_avg_calibration, sorted_calibrators_per_class, show=True)
        makeplot_density_avg(sorted_calibrators_per_class, show=True)

        plot_histogram_log_lr(h0_h1_lrs_avg_log, title='before', density=True)
        plot_histogram_log_lr(h0_h1_lrs_avg_after_log, n_bins=15, title='after', density=True)
        plot_pav(h0_h1_lrs_avg, h0_h1_lrs_avg_after, y, classes_map_updated, on_screen=True)

        pickle.dump(model, open(model_file_name, 'wb'))
        pickle.dump(calibrators_per_class, open('calibrators_per_class', 'wb'))

    else:
        model = pickle.load(open(model_file_name, 'rb'))
        calibrators_per_class = pickle.load(open('calibrators_per_class', 'rb'))

        # TODO: Why train data needed?
        X_train, y_train, y_augmented_matrix, mixture_classes_in_single_cell_type = augment_data(
            X_train,
            y_train,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

    X_mixtures, y_mixtures, y_mixtures_matrix, test_map, inv_test_map = read_mixture_data(
        n_single_cell_types - 1,
        classes_map
    )

    if retrain:
        X_augmented, y_augmented, _, _ = augment_data(
            X_train,
            y_train,
            n_single_cell_types,
            n_features,
            N_SAMPLES_PER_COMBINATION,
            classes_map,
            from_penile=from_penile
        )

        unique_augmented = np.unique(X_augmented, axis=0)
        dists_from_xmixtures_to_closest_augmented = []
        for x in tqdm(X_mixtures, 'computing distances'):
            dists_from_xmixtures_to_closest_augmented.append(np.min([np.linalg.norm(x - y) for y in unique_augmented]))
        pickle.dump(dists_from_xmixtures_to_closest_augmented, open('dists', 'wb'))
    else:
        dists_from_xmixtures_to_closest_augmented = pickle.load(open('dists', 'rb'))

    mixture_classes_in_classes_to_evaluate, classes_map_updated, y_mixtures_classes_to_evaluate_n_hot = \
        create_information_on_classes_to_evaluate(
            mixture_classes_in_single_cell_type,
            classes_map,
            class_combinations_to_evaluate,
            y_mixtures,
            y_mixtures_matrix
    )

    h0_h1_probs_mixture = model.predict_proba_per_class(
        combine_samples(X_mixtures),
        y_mixtures_classes_to_evaluate_n_hot,
        mixture_classes_in_classes_to_evaluate,
        classes_map_updated,
        MAX_LR
    )

    # transform the probabilities with the calibrated models
    h0_h1_after_calibration_mixture = calibration_transform(h0_h1_probs_mixture, classes_map_updated)

    plot_for_experimental_mixture_data(
        combine_samples(X_mixtures),
        y_mixtures,
        y_mixtures_classes_to_evaluate_n_hot,
        inv_test_map,
        classes_to_evaluate,
        mixture_classes_in_classes_to_evaluate,
        n_single_cell_types - 1,
        dists_from_xmixtures_to_closest_augmented
    )
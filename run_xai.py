#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
from functools import partial

import lime.lime_tabular
import numpy as np

import rna.settings as settings

if __name__ == "__main__":
    # Load data
    for p, priors in enumerate(settings.priors):
        data = pickle.load(open(os.path.join('models', f'data_{priors}'), 'rb'))
        label_encoder = pickle.load(open(os.path.join('models', 'label_encoder'), 'rb'))
        mlr_model = pickle.load(open(os.path.join('models', f'bin_sig_MLR_{priors}'), 'rb'))
        mlp_model = pickle.load(open(os.path.join('models', f'bin_sig_MLP_{priors}'), 'rb'))
        xgb_model = pickle.load(open(os.path.join('models', f'bin_sig_XGB_{priors}'), 'rb'))

    X, y = data.X_train_augmented, data.y_train_nhot_augmented

    # the target classes used to train the models with
    target_classes = np.array([[0., 1., 0., 0., 0., 0., 0., 1.], [0., 0., 0., 1., 0., 0., 0., 0.]])
    # the target class we will be looking at
    i_target_class = 1


    ############# find the datapoints with very wrong predictions

    mlr_predictions = np.log10(mlr_model.predict_lrs(X, target_classes, with_calibration=False))
    mlp_predictions = np.log10(mlp_model.predict_lrs(X, target_classes, with_calibration=True))
    xgb_predictions = np.log10(xgb_model.predict_lrs(X, target_classes, with_calibration=True))
    threshold = 2
    false_positives = [(i, truth, mlrp, mlpp, xgbp) for i, (truth, mlrp, mlpp, xgbp) in
                       enumerate(zip(y, mlr_predictions[:,i_target_class], mlp_predictions[:,i_target_class], xgb_predictions[:,i_target_class]))
                       if (mlrp > threshold or mlpp > threshold or xgbp > threshold) and np.inner(truth, target_classes[i_target_class]) == 0]
    false_negatives = [(i, truth, mlrp, mlpp, xgbp) for i, (truth, mlrp, mlpp, xgbp) in
                       enumerate(zip(y, mlr_predictions[:,i_target_class], mlp_predictions[:,i_target_class], xgb_predictions[:,i_target_class]))
                       if (mlrp < -threshold or mlpp < -threshold or xgbp < -threshold) and np.inner(truth, target_classes[i_target_class]) == 1]

    print(false_positives)

    # Select data point for explaining its prediction

    # for train data:
    # i=130 # XGB en MLP goede voorspelling, MLR matig
    # i=4965 # XGB schiet harder uit de bocht dan MLP en MLR, omdat eerste alleen op 1 feature focused
    # i = 3448 # MLR voorspelt verkeerd vag/menstr omdat feat 7 erin zit. andere dicht bij LR=1
    # XGB en MLP lijken wel lineair - alle gezien datapunten geven 8 10 9 ongeveer even groot
    # zelfs voor MLR geeft lime niet altijd dezelfde componenten (kan)
    for false in false_positives + false_negatives:
        i_data = false[0]
        print(i_data)
        x = X[i_data, :].reshape(1, -1)


        ############ plot lime predictions on the selected data point. Also print the model performance
        explainer = lime.lime_tabular.LimeTabularExplainer(X, discretize_continuous=False, mode='regression')

        print(f'real classes {y[i_data,:]}: {[label_encoder.classes_[j] for j, yi in enumerate(y[i_data,:]) if yi]}')
        print(f'x = {x}')
        for model, string in ((mlr_model, 'mlr'), (mlp_model, 'mlp'), (xgb_model, 'xgb')):
            print(f"{string} log LR prediction: {np.log10(model.predict_lrs(x, target_classes))}"
                  )
            fn = lambda x: np.log10(model.predict_lrs(x, target_classes=target_classes)[:,i_target_class])
            exp = explainer.explain_instance(X[i_data], fn, num_features=3, top_labels=1)
            exp.save_to_file(os.path.join('scratch',f'lime_{string}'), show_table=True, show_all=True)

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

    # Whitelist of features - list of features we can change/use when computing a counterfactual
    features_whitelist = None  # We can use all features

    # Select data point for explaining its prediction
    i=130 # XGB en MLP goede voorspelling, MLR matig
    x = X[i,:].reshape(1,-1)
    target_classes = np.array([[0., 1., 0., 0., 0., 0., 0., 1.], [0., 0., 0., 1., 0., 0., 0., 0.]])
    print(f"MLR ln LR prediction: {np.log(mlr_model.predict_lrs(x, target_classes, with_calibration=False))}"
          f"    \n  on x({x})")
    print(f'real class {y[i,:]}')
    i_target_class = 0


    explainer = lime.lime_tabular.LimeTabularExplainer(X, discretize_continuous=False, mode='regression')

    fn = lambda x: np.log(mlr_model.predict_lrs(x, target_classes=target_classes)[:,i_target_class])
    exp = explainer.explain_instance(X[i], fn, num_features=3, top_labels=1)
    exp.save_to_file(os.path.join('scratch','lime_mlr'), show_table=True, show_all=True)
    for feature, feature_val in exp.local_exp[1]:
        print(f'for feature {feature}, lime found {feature_val}, MLR has {mlr_model._classifier.coef_[i_target_class,feature]}')

    print(f"MLP ln LR prediction: {np.log(mlp_model.predict_lrs(x, target_classes, with_calibration=False))}"
          f"    on on x({x})")
    fn = lambda x: np.log(mlp_model.predict_lrs(x, target_classes=target_classes)[:,i_target_class])
    exp = explainer.explain_instance(X[i], fn, num_features=3, top_labels=1)
    exp.save_to_file(os.path.join('scratch','lime_mlp'), show_table=True, show_all=True)

    print(f"XGB ln LR prediction: {np.log(xgb_model.predict_lrs(x, target_classes, with_calibration=False))}"
          f"    on on x({x})")
    fn = lambda x: np.log(xgb_model.predict_lrs(x, target_classes=target_classes)[:,i_target_class])
    exp = explainer.explain_instance(X[i], fn, num_features=3, top_labels=1)
    exp.save_to_file(os.path.join('scratch','lime_xgb'), show_table=True, show_all=True)
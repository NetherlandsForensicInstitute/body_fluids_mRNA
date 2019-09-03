#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle

import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ceml.sklearn import generate_counterfactual
import rna.settings as settings
import numpy as np


from rna.input_output import read_mixture_data

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
    features_whitelist = None   # We can use all features

    # Select data point for explaining its prediction
    x = X[0,:].reshape(1,-1)
    print("Prediction on x: {0}".format(mlr_model.predict_lrs(x, np.array([[0., 1., 0., 0., 0., 0., 0., 1.],
                                                                          [0., 0., 0., 1., 0., 0., 0., 0.]]), with_calibration=False)))


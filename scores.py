import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier

from reimplementation import *

class ScoresMLP():

    def __init__(self):
        self._mlp = MLPClassifier()

    def fit(self, X, y):
        """
        Fit train data with MLP
        """
        results = self._mlp.fit(X, y)
        return results

    def predict(self, X):
        """
        Predict scores
        """
        y_pred = self._mlp.predict(X)
        return y_pred

    def predict_proba(self, X, labels_in_class, y_n_hot):
        y_prob = self._mlp.predict_proba(X)

        h1_h2_probs_per_class = {}
        # marginal for each single class sample
        prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
            y_prob, labels_in_class, classes_map, MAX_LR)
        for j in range(y_n_hot.shape[1]):
            cell_type = list(classes_map.keys())[list(classes_map.values()).index(j)]
            # get the probability per single class sample
            total_proba = prob_per_class[:, j]
            if sum(total_proba) > 0:
                probas_without_cell_type = total_proba[y_n_hot[:, j] == 0]
                probas_with_cell_type = total_proba[y_n_hot[:, j] == 1]
                h1_h2_probs_per_class[cell_type] = (probas_with_cell_type, probas_without_cell_type)

        return h1_h2_probs_per_class


if __name__ == '__main__':
    model = ScoresMLP()
    X_augmented_train = pickle.load(open('X_augmented_train', 'rb'))
    y_augmented_train = pickle.load(open('y_augmented_train', 'rb'))
    X_augmented_test = pickle.load(open('X_augmented_test', 'rb'))

    model.fit(X_augmented_train, y_augmented_train)
    test = model.predict(X_augmented_test)

